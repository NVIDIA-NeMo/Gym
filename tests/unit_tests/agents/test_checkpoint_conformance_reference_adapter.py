# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The harness checkpoint conformance kit, run against the smallest adapter that meets its contract.

The adapter's harness is a shell loop that waits for a file. It has no conversation, so a relaunch starts it again;
everything else is what a real adapter does: the session borrows the kit's sandbox, each launch runs under the real
process supervisor through ``run_with_checkpoint_pauses``, and the session is exported, restored and re-bound.
"""

import asyncio
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, Request
from pydantic import JsonValue

from nemo_gym.agents.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.agents.testing.checkpoint_conformance import (
    HarnessFixture,
    check_harness_checkpointing,
    process_state,
)
from nemo_gym.agents.utils.run_with_checkpoint_pauses import LaunchFinished, run_with_checkpoint_pauses
from nemo_gym.agents.utils.sandbox_session import HarnessPauseResult, SandboxCommand, SandboxSession
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.server_runtime._checkpoint.agent import RestoredAgentSession
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory.trajectory_record import AgentObservationBundle


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None, reason="Linux /proc and bash are required"
)

_CLOSE_TIMEOUT_S = 30.0


@dataclass
class _Session(AgentSessionState):
    # ``None`` after a restore, until the replacement attempt's seed re-binds the session to its sandbox.
    session: SandboxSession[None] | None
    launch_count: int = 0
    completed_activation: int = 0
    completed_reply: dict[str, JsonValue] | None = None
    restored: dict[str, JsonValue] | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None


class _ReferenceAgent(SimpleResponsesAPIAgent):
    """One activation per session, in a sandbox the session borrows; the harness is ``harness``, an argv."""

    checkpoint_mode = "exported"
    provider: Any = None
    harness: Any = None
    session_root: Any = None

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise NotImplementedError

    def _continuation_close_timeout(self) -> float:
        return _CLOSE_TIMEOUT_S

    async def _connect(self, body: AgentSeedSessionRequest) -> AsyncSandbox:
        return await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=self.provider)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> _Session:
        session = SandboxSession(
            sandbox=await self._connect(body),
            session_dir=f"{self.session_root}/{uuid4().hex}",
            workdir=body.sandbox_access.workdir,
            harness="reference",
            pausable=True,
        )
        return _Session(request=body, session=session)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        state = self._require_agent_session(session_id)
        assert isinstance(state, _Session)
        participant = self.checkpoint_participant
        participant.check_session(session_id, EpisodeId.from_capture_key(request.path_params["rollout_id"]))
        index = self._activation_index(request)
        if index == state.completed_activation and state.completed_reply is not None:
            participant.answered_from_saved_reply(session_id, state.request.episode_id)
            return NeMoGymResponse.model_validate(state.completed_reply)
        if state.session is None:
            raise AgentSessionConflictError("the session was restored and not re-bound: seed it again first")
        if state.task is None:
            state.task = asyncio.create_task(self._activation(state, index))
        return await asyncio.shield(state.task)

    async def _activation(self, state: _Session, index: int) -> NeMoGymResponse:
        session = state.session

        async def stage() -> SandboxCommand:
            return SandboxCommand(argv=list(self.harness))

        async def collect() -> None:
            return None

        async with self.checkpoint_participant.activation(
            state.request.agent_session_id, state.request.episode_id, index=index
        ) as activation:
            while True:
                if state.launch_count:
                    # An earlier launch was stopped, or ran before the checkpoint this session was restored from.
                    await session.begin_activation(
                        session_dir=f"{session.root_dir}/launch-{state.launch_count + 1}", timeout=_CLOSE_TIMEOUT_S
                    )
                state.launch_count += 1
                launch = state.launch_count
                outcome = await run_with_checkpoint_pauses(
                    session,
                    activation,
                    launch=lambda: session.execute(
                        stage_activation=stage,
                        collect=collect,
                        timeout=300,
                        close_timeout=_CLOSE_TIMEOUT_S,
                        pause_allowance_s=600,
                    ),
                    on_parked=lambda: {"launch": launch},
                    stop_timeout=_CLOSE_TIMEOUT_S,
                )
                if isinstance(outcome, LaunchFinished):
                    break
            response = NeMoGymResponse(
                id=f"resp-{uuid4().hex}",
                created_at=0,
                model="reference",
                object="response",
                output=[],
                tool_choice="auto",
                parallel_tool_calls=True,
                tools=[],
            )
            # Recorded before the activation ends, so a checkpoint exports the session with its reply.
            state.completed_activation = index
            state.completed_reply = response.model_dump(mode="json")
            return response

    async def export_agent_sessions(
        self, session_keys: list[str], *, checkpoint_id: str
    ) -> dict[str, dict[str, JsonValue]]:
        exported: dict[str, dict[str, JsonValue]] = {}
        for key in session_keys:
            state = self._session_state(key)
            assert isinstance(state, _Session)
            exported[key] = dict(state.restored or {}) or {
                "request": state.request.model_dump(mode="json"),
                "session": dict(state.session.checkpoint_state()),
                "launch_count": state.launch_count,
                "completed_activation": state.completed_activation,
                "completed_reply": state.completed_reply,
            }
        return exported

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        for restored in sessions:
            saved = restored.session
            request = AgentSeedSessionRequest.model_validate(
                saved["request"] | {"episode_id": restored.episode_id.model_dump(mode="json")}
            )
            state = _Session(
                request=request,
                session=None,
                launch_count=saved["launch_count"],
                completed_activation=saved["completed_activation"],
                completed_reply=saved["completed_reply"],
                restored=dict(saved),
            )
            self._install_restored_session(restored.session_key, state)

    async def _rebind_restored_agent_session_state(
        self, state: AgentSessionState, body: AgentSeedSessionRequest
    ) -> AgentSessionState:
        assert isinstance(state, _Session)
        if state.session is None:
            state.session = SandboxSession.rebind(
                await self._connect(body),
                state.restored["session"],
                workdir=body.sandbox_access.workdir,
                harness="reference",
                pausable=True,
            )
            state.restored = None
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, _Session)
        if state.session is not None:
            # Stops a running harness and releases the borrowed connection, then ends the activation.
            await state.session.close(timeout=_CLOSE_TIMEOUT_S)
            if state.task is not None:
                state.task.cancel()
                await asyncio.wait({state.task}, timeout=_CLOSE_TIMEOUT_S)
                if state.task.done() and not state.task.cancelled():
                    state.task.exception()
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(schema_version="2.0", source="reference"),
        )


class _ReferenceFixture(HarnessFixture):
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.control = tmp_path / "control"
        (self.control / "pids").mkdir(parents=True)
        (self.control / "launches").write_text("")
        (tmp_path / "task").mkdir()
        self.script = tmp_path / "harness.sh"
        # A launch records its process, then that it started, and runs until the fixture lets launches end.
        self.script.write_text(
            f': > "{self.control}/pids/$$"\n'
            f'echo launch >> "{self.control}/launches"\n'
            f'while [ ! -e "{self.control}/finish" ]; do sleep 0.05; done\n'
        )

    def make_agent(self, global_config: dict[str, JsonValue]) -> _ReferenceAgent:
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = global_config
        agent = _ReferenceAgent(
            config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="reference"), server_client=client
        )
        agent.provider = self.provider
        agent.harness = ["bash", str(self.script)]
        agent.session_root = str(self.tmp_path / "sessions")
        return agent

    def seed(self, agent_session_id: str, attempt: int) -> dict[str, JsonValue]:
        return AgentSeedSessionRequest(
            agent_session_id=agent_session_id,
            episode_id=EpisodeId(rollout_id="r", attempt=attempt),
            task_id=TaskId(taskset="reference", task_id="task"),
            sandbox_access={
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "sandbox",
                    "descriptor": {"sandbox_id": "local"},
                },
                "workdir": str(self.tmp_path / "task"),
            },
        ).model_dump(mode="json")

    def activation(self) -> dict[str, JsonValue]:
        return {"input": "task"}

    def started(self) -> int:
        return len((self.control / "launches").read_text().splitlines())

    def harness_pids(self) -> list[int]:
        pids = (int(path.name) for path in (self.control / "pids").iterdir())
        return [pid for pid in pids if process_state(pid) not in (None, "Z", "X")]

    def finish(self) -> None:
        (self.control / "finish").touch()

    def hold(self) -> None:
        (self.control / "finish").unlink(missing_ok=True)

    def break_next_pause(self, agent: SimpleResponsesAPIAgent) -> None:
        original = SandboxSession.pause_harness

        async def unconfirmed(session: SandboxSession, **kwargs: Any) -> HarnessPauseResult:
            self.monkeypatch.setattr(SandboxSession, "pause_harness", original)
            return HarnessPauseResult(status="unconfirmed", error="injected: a process stuck in disk I/O")

        self.monkeypatch.setattr(SandboxSession, "pause_harness", unconfirmed)

    def crash_recovery_assertion(self) -> Callable[[dict[str, JsonValue]], None]:
        # This harness keeps no conversation: what recovery can be held to is one relaunch and a new reply.
        launches = self.started()
        self.recovered: list[str] = []

        def continued(reply: dict[str, JsonValue]) -> None:
            assert self.started() == launches + 1, "recovery relaunches the harness once"
            self.recovered.append(str(reply["id"]))

        return continued


async def test_a_minimal_adapter_conforms_to_the_harness_checkpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _ReferenceFixture(tmp_path, monkeypatch)
    try:
        await check_harness_checkpointing(fixture, tmp_path / "state")
    finally:
        fixture.finish()
    assert fixture.harness_pids() == [] and fixture.provider.processes() == []
    # The kit asked for the crash check and gave it the reply of the activation that recovered.
    assert len(fixture.recovered) == 1 and fixture.recovered[0].startswith("resp-")


async def test_the_kit_fails_an_adapter_that_relaunches_instead_of_pausing_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _ReferenceFixture(tmp_path, monkeypatch)

    async def unconfirmed(session: SandboxSession, **kwargs: Any) -> HarnessPauseResult:
        return HarnessPauseResult(status="unconfirmed", error="this adapter cannot pause")

    monkeypatch.setattr(SandboxSession, "pause_harness", unconfirmed)
    try:
        with pytest.raises(AssertionError, match="stopped while the checkpoint is open"):
            await check_harness_checkpointing(fixture, tmp_path / "state")
    finally:
        fixture.finish()
    # Also after a failed check, nothing the kit or the adapter started is left running.
    assert fixture.harness_pids() == [] and fixture.provider.processes() == []
