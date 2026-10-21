# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a black-box agent harness in a lent sandbox as one long, checkpointable step.

An agent such as OpenCode seeds a session on its resources server, which lends it a sandbox, runs the harness
there as one long shell command, and verifies what the harness did. :class:`SandboxedHarnessAgent` writes the
checkpoint logic of such an agent once; the harness-specific parts are a :class:`HarnessAdapter`.

The run records a bookmark (a ``LegacyRun`` boundary) before each step: ``seed``, ``harness``, ``verify``,
``return``. When a checkpoint asks the run to park during the harness step, the launch is paused if the adapter
pauses at checkpoints, and stopped completely otherwise (``nemo_gym._checkpoint.harness``). Either way the run
records a bookmark that continues the harness step with ``resume`` set. A paused launch parks at that bookmark and
continues in place once the checkpoint opens. A stopped one parks after the launch, and the adapter continues the
harness's saved session in a relaunch after the resume. After a crash the replacement attempt relaunches from the
bookmark in both cases. The session lives on the sandbox's disk, so the owner's snapshot carries it into a fork.

The resources server owns the sandbox. After a restore it is a fork of the checkpoint's snapshot under a new id,
so the sandbox a bookmark names is only a hint: before every launch the run asks ``/sandbox_access`` for the
current one.

A run is **checkpointed** when checkpointing is on and the run is not a restart. A checkpointed run never stops
the sandbox: the owner may still export the session, and ends it at verify. If it leaves without a verdict it
closes the session through the owner (``POST /close_session`` with an empty body). Every other run stops the
sandbox when it ends, as an agent does that owns its sandbox or whose lender cannot capture it.
"""

import asyncio
import logging
import time
from contextlib import AbstractAsyncContextManager, AsyncExitStack, nullcontext
from dataclasses import dataclass
from typing import Annotated, Any, Awaitable, Callable, ClassVar, Generic, Literal, Optional, Protocol, TypeVar, Union

from anyio import CancelScope, move_on_after
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from nemo_gym._checkpoint.agent import LegacyRun, RestoredAgentSession, require_rollout
from nemo_gym._checkpoint.harness import HarnessInterrupted, ResumeDecision, run_until_park_requested
from nemo_gym._checkpoint.steps import StepMode, seed_restarts, seed_verify_mode
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.episode_types import EpisodeId
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import raise_for_status


LOGGER = logging.getLogger(__name__)

# Session keys of legacy /run episodes in the checkpoint participant; everything a restore needs is in the bookmark.
LEGACY_SESSION_PREFIX = "run:"
BOOKMARK_VERSION = 1
# How long a checkpointed run that leaves without a verdict waits for the owner to close its session. The close
# waits out a checkpoint that is open, as every resources request does.
_CLOSE_TIMEOUT_S = 120.0

OutcomeT = TypeVar("OutcomeT", bound=BaseModel)


def cookie_values(cookies: Any) -> dict[str, str]:
    """Flatten a plain dict or an aiohttp cookie jar of morsels into name-to-value pairs."""
    return {str(name): str(getattr(value, "value", value)) for name, value in (cookies or {}).items()}


class LentSandbox(BaseModel):
    """The sandbox an owner lends, from its seed reply or ``/sandbox_access``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: dict[str, JsonValue]
    workdir: Optional[str] = None

    @property
    def sandbox_id(self) -> Optional[str]:
        sandbox_id = self.descriptor.get("sandbox_id")
        return str(sandbox_id) if sandbox_id is not None else None

    @classmethod
    def from_seed_reply(cls, reply: Any) -> Optional["LentSandbox"]:
        """The sandbox a seed reply names (``sandbox_descriptor``, else ``sandbox_handle``); ``None`` if none."""
        if not isinstance(reply, dict):
            return None
        workdir = reply.get("workdir") if isinstance(reply.get("workdir"), str) else None
        if isinstance(reply.get("sandbox_descriptor"), dict):
            return cls(descriptor=reply["sandbox_descriptor"], workdir=workdir)
        if reply.get("sandbox_handle"):
            return cls(descriptor={"sandbox_id": str(reply["sandbox_handle"])}, workdir=workdir)
        return None


class LaunchRecord(BaseModel):
    """One launch of the harness: whether it continued a saved session, why, how long it ran, how it ended, and how
    many checkpoints paused it."""

    resumed: bool
    reason: Optional[str] = None
    seconds: float
    ended: Literal["finished", "interrupted", "failed"]
    pauses: int = 0


class _Bookmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = BOOKMARK_VERSION


class SeedBookmark(_Bookmark):
    next: Literal["seed"] = "seed"


class HarnessBookmark(_Bookmark):
    next: Literal["harness"] = "harness"
    cookies: dict[str, str]
    verify_mode: StepMode
    # A hint: the run asks the owner for the current sandbox before every launch.
    sandbox: Optional[LentSandbox]
    # The adapter's per-rollout state, pinned across launches, such as where the harness keeps its session.
    state: dict[str, JsonValue]
    # Continue the saved session instead of starting the task.
    resume: bool
    launches: list[LaunchRecord]


class VerifyBookmark(_Bookmark):
    next: Literal["verify"] = "verify"
    cookies: dict[str, str]
    verify_mode: StepMode
    outcome: dict[str, JsonValue]
    launches: list[LaunchRecord]
    # What a prepared verification captured (such as the patch) once the owner ended the session: grading it
    # is replayable, so a checkpoint does not wait for it and a restore grades it again.
    prepared: Optional[dict[str, JsonValue]] = None


class ReturnBookmark(_Bookmark):
    next: Literal["return"] = "return"
    result: dict[str, JsonValue]
    outcome: dict[str, JsonValue]
    launches: list[LaunchRecord]


Bookmark = Annotated[Union[SeedBookmark, HarnessBookmark, VerifyBookmark, ReturnBookmark], Field(discriminator="next")]
_BOOKMARK: TypeAdapter[Bookmark] = TypeAdapter(Bookmark)


class BookmarkError(ValueError):
    """A continuation's bookmark is of an unknown version or malformed: the attempt fails rather than guess."""


def parse_bookmark(continuation: Optional[dict[str, Any]]) -> Bookmark:
    """The bookmark a continuation carries; ``seed`` for a run that starts from its input."""
    if not continuation:
        return SeedBookmark()
    version = continuation.get("v")
    if version != BOOKMARK_VERSION:
        raise BookmarkError(f"harness bookmark version {version!r} is not supported; this agent reads version 1")
    try:
        return _BOOKMARK.validate_python(continuation)
    except ValidationError as error:
        raise BookmarkError(f"malformed harness bookmark: {error}") from error


class LaunchContext:
    """One launch of the harness, as the adapter sees it.

    ``state`` is the adapter's per-rollout state (see :meth:`HarnessAdapter.new_state`); ``resume`` says to
    continue the saved session in it. ``park``, set for a checkpointed run whose adapter pauses at checkpoints,
    records the run's bookmark and returns once the checkpoint opens.
    """

    def __init__(
        self,
        sandbox: AsyncSandbox,
        state: dict[str, JsonValue],
        *,
        resume: bool,
        legacy_run: Optional[LegacyRun],
        park: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self.sandbox = sandbox
        self.state = state
        self.resume = resume
        self.park = park
        self._legacy_run = legacy_run

    async def run(self, command: str, *, timeout_s: Optional[float]) -> SandboxExecResult:
        """Run the harness's long command so that a checkpoint pauses it, or stops it and everything it started.

        With ``park`` set, a checkpoint pauses the command and it continues once the checkpoint opens. Raises
        :class:`~nemo_gym._checkpoint.harness.HarnessInterrupted` when a checkpoint stopped it before it ended.
        Run short commands, such as reading the session, with ``sandbox.exec``.
        """
        return await run_until_park_requested(
            self.sandbox, command, self._legacy_run, timeout_s=timeout_s, park=self.park
        )

    async def wait_park_requested(self) -> None:
        """Return once a checkpoint asks this run to park; never for a run that is not checkpointed.

        For an adapter that stops its harness its own way: it raises ``HarnessInterrupted`` once it has.
        """
        if self._legacy_run is None:
            await asyncio.Event().wait()
        else:
            await self._legacy_run.wait_park_requested()


class HarnessAdapter(Protocol[OutcomeT]):
    """What a harness needs to run checkpointed under :class:`SandboxedHarnessAgent`."""

    #: What the run keeps of a finished launch, for verify and the agent's response; stored in the bookmark.
    outcome_model: type[OutcomeT]

    def new_state(self) -> dict[str, JsonValue]:
        """Fresh per-rollout state, such as a new folder for the harness's session on the sandbox's disk."""
        ...

    async def connect(self, lent: Optional[LentSandbox]) -> AsyncSandbox:
        """A client of the lent sandbox; with none lent, a sandbox of the agent's own."""
        ...

    async def resume_decision(self, sandbox: AsyncSandbox, state: dict[str, JsonValue]) -> ResumeDecision:
        """Whether the saved session in ``state`` can be continued: only one that carries the task."""
        ...

    async def launch(self, launch: LaunchContext) -> OutcomeT:
        """Launch the harness, through ``launch.run`` for its long command, and read what it did."""
        ...

    async def verify(self, cookies: dict[str, str], outcome: OutcomeT) -> dict[str, Any]:
        """Grade the outcome with the resources server; returns the ``/verify`` reply."""
        ...


class PausesAtCheckpoints(Protocol):
    """Optional for a :class:`HarnessAdapter`: pause the harness at a checkpoint instead of stopping it.

    A paused launch continues in place once the checkpoint opens, so the harness is not relaunched. A launch whose
    pause cannot be confirmed is stopped as before.
    """

    def pauses_at_checkpoints(self) -> bool:
        """Whether this run pauses its harness at checkpoints."""
        ...


class PreparedVerification(Protocol[OutcomeT]):
    """Optional for a :class:`HarnessAdapter`: verification in two steps, so a checkpoint never waits for grading.

    ``prepare_verification`` runs as a wait step: it captures what grading needs and ends the session, which
    frees the sandbox. ``verify_prepared`` grades that capture as a replay step.
    """

    def prepares_verification(self) -> bool:
        """Whether this run verifies in two steps."""
        ...

    async def prepare_verification(self, cookies: dict[str, str], outcome: OutcomeT) -> dict[str, JsonValue]:
        """Capture what grading needs and end the session; a retry returns the same capture."""
        ...

    async def verify_prepared(
        self, cookies: dict[str, str], outcome: OutcomeT, prepared: dict[str, JsonValue]
    ) -> dict[str, Any]:
        """Grade ``prepared``; returns the grading reply."""
        ...


@dataclass(frozen=True)
class HarnessRunResult(Generic[OutcomeT]):
    """A finished run: the ``/verify`` reply, the outcome of the harness step, and every launch."""

    result: dict[str, Any]
    outcome: OutcomeT
    launches: list[LaunchRecord]


class SandboxedHarnessAgent(SimpleResponsesAPIAgent):
    """An agent whose legacy ``/run`` runs a harness in a lent sandbox as one long, checkpointable step.

    The config must name the ``resources_server`` that lends the sandbox. A subclass's ``run`` calls
    :meth:`run_harness` with its adapter and builds its response from the result.
    """

    checkpoint_sessions_supported: ClassVar[bool] = True

    # -- partial-rollout checkpoints: a legacy /run keeps everything a restore needs in its bookmark ------------

    async def export_agent_sessions(
        self, session_keys: list[str], *, checkpoint_id: str
    ) -> dict[str, dict[str, JsonValue]]:
        for session_key in session_keys:
            if not session_key.startswith(LEGACY_SESSION_PREFIX):
                raise TypeError(f"{type(self).__name__} has no session {session_key!r}: it runs legacy /run only")
        return {session_key: {} for session_key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        for session in sessions:
            if not session.session_key.startswith(LEGACY_SESSION_PREFIX):
                raise TypeError(f"{type(self).__name__} cannot restore session {session.session_key!r}")

    async def retire_agent_session(self, session_key: str) -> None:
        pass

    # -- the run -----------------------------------------------------------------------------------------------

    async def run_harness(
        self,
        request: Request,
        body: BaseRunRequest,
        adapter: HarnessAdapter[OutcomeT],
        *,
        slot: Optional[AbstractAsyncContextManager[Any]] = None,
    ) -> HarnessRunResult[OutcomeT]:
        """Seed, run the harness until it finishes, and verify; checkpointed when checkpointing is on.

        ``slot``, such as the agent's concurrency semaphore, is held for the whole run. With checkpointing on, the
        run is registered before it waits for the slot: parked runs keep their slots, so a run queued behind them
        must be exported at its bookmark instead of staying invisible to a checkpoint that cannot wait for it.
        """
        participant = self.checkpoint_participant
        if participant is None:
            async with slot if slot is not None else nullcontext():
                return await _HarnessRun(self, request, body, adapter, None).drive()
        episode_id = EpisodeId.from_capture_key(require_rollout(self.rollout_id_from_run(body)))
        async with participant.legacy_run(f"{LEGACY_SESSION_PREFIX}{episode_id.rollout_id}", episode_id) as run:
            async with AsyncExitStack() as held:
                if slot is not None:
                    # Nothing of the task has run yet, so the bookmark it continues from is a safe point.
                    bookmark = run.continuation
                    if bookmark is None:
                        bookmark = SeedBookmark().model_dump(mode="json")
                    await run.boundary(bookmark)
                    # Waiting for a process-local slot is safe to replay from that bookmark.
                    async with run.step("replay"):
                        await held.enter_async_context(slot)
                    # A slot may come free while a checkpoint has admission closed: park before any task work.
                    await run.boundary(bookmark)
                return await _HarnessRun(self, request, body, adapter, run).drive()

    async def refresh_lent_sandbox(
        self, cookies: dict[str, str], hint: Optional[LentSandbox]
    ) -> Optional[LentSandbox]:
        """The session's current sandbox from the resources server; ``hint`` if it lends none (404).

        Any other failure raises: after a restore the hint names the superseded sandbox, and running there would be
        scored as the model's failure.
        """
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/sandbox_access", cookies=cookies
        )
        if response.status == 404:
            return hint
        await raise_for_status(response)
        access = SandboxAccess.model_validate(await response.json())
        descriptor = getattr(access.connection, "descriptor", None)
        if not isinstance(descriptor, dict):
            raise ValueError(
                f"/sandbox_access named a {access.connection.kind!r} connection, not a sandbox descriptor"
            )
        return LentSandbox(descriptor=descriptor, workdir=access.workdir)


class _HarnessRun(Generic[OutcomeT]):
    """The stage machine of one run: seed, harness (a loop of launches), verify, return."""

    def __init__(
        self,
        agent: SandboxedHarnessAgent,
        request: Request,
        body: BaseRunRequest,
        adapter: HarnessAdapter[OutcomeT],
        legacy_run: Optional[LegacyRun],
    ) -> None:
        self.agent = agent
        self.request = request
        self.body = body
        self.adapter = adapter
        self.legacy_run = legacy_run
        self.rollout_id = agent.rollout_id_from_run(body)
        self.sandbox: Optional[AsyncSandbox] = None
        # While the run waits at a bookmark: a checkpoint may be exporting it, and the sandbox is the checkpoint's.
        self.parked = False
        # The owner snapshots the sandbox: checkpointing is on and the run is not a restart. Such a run never stops
        # the sandbox; the owner ends the session at verify, or when the run closes it.
        self.checkpointed = legacy_run is not None
        # The session cookies once seeded; a checkpointed run that leaves without a verdict closes the session.
        self.cookies: Optional[dict[str, str]] = None
        self.verified = False
        # A prepared verification ended the session: there is nothing left to close.
        self.session_ended = False

    async def record(self, bookmark: Bookmark) -> None:
        if self.legacy_run is None:
            return
        # Left set if the wait ends in an exception, such as the caller going away while the run is parked.
        self.parked = True
        await self.legacy_run.boundary(bookmark.model_dump(mode="json"))
        self.parked = False

    def step(self, mode: StepMode) -> AbstractAsyncContextManager[None]:
        return self.legacy_run.step(mode) if self.legacy_run is not None else nullcontext()

    async def drive(self) -> HarnessRunResult[OutcomeT]:
        bookmark = parse_bookmark(self.legacy_run.continuation if self.legacy_run is not None else None)
        outcome: Optional[OutcomeT] = None
        try:
            if isinstance(bookmark, SeedBookmark):
                bookmark = await self.seed()
            if isinstance(bookmark, (HarnessBookmark, VerifyBookmark)):
                self.cookies = bookmark.cookies
            while isinstance(bookmark, HarnessBookmark):
                bookmark, outcome = await self.launch_once(bookmark)
            if isinstance(bookmark, VerifyBookmark):
                if outcome is None:
                    outcome = self.adapter.outcome_model.model_validate(bookmark.outcome)
                prepares = getattr(self.adapter, "prepares_verification", None)
                if bookmark.prepared is None and prepares is not None and prepares():
                    await self.record(bookmark)
                    # Short and state-changing (it ends the session), so a checkpoint waits for it.
                    async with self.step("wait"):
                        prepared = await self.adapter.prepare_verification(bookmark.cookies, outcome)
                    bookmark = bookmark.model_copy(update={"prepared": prepared})
                self.session_ended = bookmark.prepared is not None
                await self.record(bookmark)
                if bookmark.prepared is not None:
                    async with self.step("replay"):
                        result = await self.adapter.verify_prepared(bookmark.cookies, outcome, bookmark.prepared)
                else:
                    async with self.step(bookmark.verify_mode):
                        result = await self.adapter.verify(bookmark.cookies, outcome)
                self.verified = True
                bookmark = ReturnBookmark(result=result, outcome=bookmark.outcome, launches=bookmark.launches)
                # Recorded so that a restore never runs a state-changing verification twice.
                await self.record(bookmark)
        finally:
            await self.cleanup()
        return HarnessRunResult(
            result=dict(bookmark.result),
            outcome=outcome if outcome is not None else self.adapter.outcome_model.model_validate(bookmark.outcome),
            launches=bookmark.launches,
        )

    async def seed(self) -> HarnessBookmark:
        await self.record(SeedBookmark())
        agent = self.agent
        # Seeds are idempotent: a replacement that re-seeds a restored session gets the same sandbox.
        async with self.step("replay"):
            reply = await agent.server_client.post(
                server_name=agent.config.resources_server.name,
                url_path="/seed_session",
                json=self.body.model_dump(),
                cookies=dict(self.request.cookies),
            )
            await raise_for_status(reply)
            cookies = cookie_values(self.request.cookies) | cookie_values(reply.cookies)
            headers = getattr(reply, "headers", None)
            lent = LentSandbox.from_seed_reply(await reply.json())
        if self.legacy_run is not None and (seed_restarts(headers) or lent is None):
            # Nothing saves the sandbox the harness runs in: the resources server cannot capture its session, or
            # the sandbox is the agent's own. The run is a restart: never parked, and after a crash it starts over.
            await self.legacy_run.mark_restart()
            self.checkpointed = False
        return HarnessBookmark(
            cookies=cookies,
            verify_mode=seed_verify_mode(headers),
            sandbox=lent,
            state=self.adapter.new_state(),
            resume=False,
            launches=[],
        )

    async def launch_once(
        self, bookmark: HarnessBookmark
    ) -> tuple[Union[HarnessBookmark, VerifyBookmark], Optional[OutcomeT]]:
        await self.record(bookmark)
        await self.disconnect()
        lent = bookmark.sandbox
        if self.legacy_run is not None:
            lent = await self.agent.refresh_lent_sandbox(bookmark.cookies, lent)
        self.sandbox = await self.adapter.connect(lent)
        state, resume, reason = bookmark.state, bookmark.resume, None
        if resume:
            decision = await self.adapter.resume_decision(self.sandbox, state)
            (LOGGER.info if decision.resume else LOGGER.warning)(
                "Harness relaunch of %s: %s", self.rollout_id, decision
            )
            reason = decision.reason
            if not decision.resume:
                # Continuing would open an empty session and the model would never see the task: start over with
                # the task, in fresh state.
                state, resume = self.adapter.new_state(), False
        bookmark = bookmark.model_copy(update={"sandbox": lent, "state": state, "resume": resume})
        if self.legacy_run is not None and self.legacy_run.park_requested:
            # A checkpoint asked this run to park before the harness started: park at the bookmark instead of
            # starting the harness only to stop it during its install.
            return bookmark, None
        started = time.monotonic()
        pauses = 0

        async def park() -> None:
            # The launch is paused. Should the process crash before it continues, the replacement attempt continues
            # the saved session from this bookmark, as after a stop.
            nonlocal pauses
            pauses += 1
            record = LaunchRecord(
                resumed=resume,
                reason=reason,
                seconds=time.monotonic() - started,
                ended="interrupted",
                pauses=pauses,
            )
            await self.record(bookmark.model_copy(update={"resume": True, "launches": [*bookmark.launches, record]}))

        pauses_at_checkpoints = getattr(self.adapter, "pauses_at_checkpoints", None)
        pausing = self.legacy_run is not None and pauses_at_checkpoints is not None and pauses_at_checkpoints()
        launch = LaunchContext(
            self.sandbox, state, resume=resume, legacy_run=self.legacy_run, park=park if pausing else None
        )
        async with self.step("wait"):
            try:
                outcome = await self.adapter.launch(launch)
            except HarnessInterrupted as stopped:
                # Stopped; the next bookmark parks this run until the resume, and continues the session in the
                # replacement attempt after a crash.
                LOGGER.info("Harness of %s stopped for a checkpoint: %s", self.rollout_id, stopped)
                record = LaunchRecord(
                    resumed=resume,
                    reason=reason,
                    seconds=time.monotonic() - started,
                    ended="interrupted",
                    pauses=pauses,
                )
                return bookmark.model_copy(update={"resume": True, "launches": [*bookmark.launches, record]}), None
        record = LaunchRecord(
            resumed=resume, reason=reason, seconds=time.monotonic() - started, ended="finished", pauses=pauses
        )
        verify = VerifyBookmark(
            cookies=bookmark.cookies,
            verify_mode=bookmark.verify_mode,
            outcome=outcome.model_dump(mode="json"),
            launches=[*bookmark.launches, record],
        )
        return verify, outcome

    async def disconnect(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        with CancelScope(shield=True):
            try:
                await sandbox.disconnect()
            except Exception:
                LOGGER.warning("Failed to release the sandbox client of %s", self.rollout_id, exc_info=True)

    async def cleanup(self) -> None:
        """Leave the sandbox as the run's state says.

        - Parked at a bookmark: the sandbox is the checkpoint's; only the client is released.
        - Checkpointed: never stop the sandbox, which the owner may still export (and stopping it under the owner
          failed the next commit). The launch is already dead: it ended, or was stopped or killed. Without a
          verdict the run closes its session through the owner, which stops the sandbox.
        - Every other run (a restart, checkpointing off, a lender that cannot capture the session, the agent's own
          sandbox): stop the sandbox, or the harness would keep running in it.
        """
        if self.parked:
            await self.disconnect()
            return
        if self.checkpointed:
            if not self.verified and not self.session_ended and self.cookies is not None:
                await self.close_session()
            await self.disconnect()
            return
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        # A server cancels its handler when the caller disconnects, and the cancellation is re-delivered at every
        # await until the handler exits: shielded, or the stop would itself be cancelled.
        with CancelScope(shield=True):
            try:
                await sandbox.stop()
            except Exception:
                LOGGER.warning("Failed to stop the sandbox of %s", self.rollout_id, exc_info=True)

    async def close_session(self) -> None:
        """Ask the owner to end the session (``POST /close_session`` with an empty body); a failure is logged, and
        the owner's retire or the provider's lease frees the sandbox."""
        agent = self.agent
        with move_on_after(_CLOSE_TIMEOUT_S, shield=True) as scope:
            try:
                response = await agent.server_client.post(
                    server_name=agent.config.resources_server.name,
                    url_path="/close_session",
                    json={},
                    cookies=self.cookies,
                )
                await raise_for_status(response)
                return
            except Exception:
                LOGGER.warning("Failed to close the resources session of %s", self.rollout_id, exc_info=True)
                return
        if scope.cancelled_caught:
            LOGGER.warning("Closing the resources session of %s timed out", self.rollout_id)
