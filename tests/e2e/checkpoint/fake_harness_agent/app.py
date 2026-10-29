# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sandboxed-harness agent for the checkpoint e2e suite, shaped like the OpenCode agent.

Its sessions borrow the sandbox the resources server lends, and run a harness there under the real process supervisor
with ``run_until_parked``: a checkpoint pauses the harness in place, a pause that fails stops it and the activation
relaunches it, and after a crash the replacement attempt's seed re-binds the restored session to the forked sandbox.
The harness makes one model call per step through the policy model server and writes one line per step to ``notes``
in the sandbox's working directory; a relaunch continues after the last line, as an exact resume does.

``fail_pauses`` makes that many pauses fail to confirm. ``finish_on_park`` lets the harness finish as the park arrives.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from shlex import quote
from time import time
from typing import Any, ClassVar, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import JsonValue


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_sandbox"))
import local_sandbox_provider  # noqa: E402, F401  (registers the local_dir providers)

from nemo_gym._checkpoint.agent import Activation, RestoredAgentSession  # noqa: E402
from nemo_gym.agent_utils.run_until_parked import Finished, run_until_parked  # noqa: E402
from nemo_gym.agent_utils.sandbox_session import HarnessPauseResult, SandboxCommand, SandboxSession  # noqa: E402
from nemo_gym.agent_utils.supervisor_client import CLEANUP_RECEIPT_FILE  # noqa: E402
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse  # noqa: E402
from nemo_gym.base_responses_api_agent import (  # noqa: E402
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef  # noqa: E402
from nemo_gym.episode_types import EpisodeId  # noqa: E402
from nemo_gym.global_config import get_global_config_dict  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponse  # noqa: E402
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap  # noqa: E402
from nemo_gym.sandbox import AsyncSandbox, create_provider  # noqa: E402
from nemo_gym.sandbox.access import DirectSandboxConnection  # noqa: E402
from nemo_gym.sandbox.config import resolve_provider_config  # noqa: E402
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint  # noqa: E402


HARNESS = """import json, os, pathlib, signal, sys, time, urllib.request
signal.signal(signal.SIGTERM, lambda *_: os._exit(143))
workdir, steps, delay, base_url, release = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
notes = pathlib.Path(workdir) / "notes"
done = len(notes.read_text().splitlines()) if notes.exists() else 0
for index in range(done, steps):
    body = json.dumps({"input": [{"role": "user", "content": f"step {index}"}]}).encode()
    request = urllib.request.Request(base_url + "/responses", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        response.read()
    with notes.open("a") as file:
        file.write(f"step-{index}\\n")
    time.sleep(delay)
# With a release file, the harness waits for it after its last step, as one that finishes as a park arrives.
while release and not pathlib.Path(release).exists():
    time.sleep(0.05)
"""


class FakeHarnessAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    steps: int = 4
    step_delay_s: float = 0.2
    fail_pauses: int = 0
    finish_on_park: bool = False
    close_timeout_s: float = 20.0


@dataclass
class FakeHarnessSession(AgentSessionState):
    session: Optional[SandboxSession[None]]
    root: str
    workdir: str
    launch_count: int = 0
    completed_activation: int = 0
    completed_reply: Optional[dict[str, JsonValue]] = None
    observations: Optional[dict[str, JsonValue]] = None
    restored: Optional[dict[str, JsonValue]] = None
    task: Optional[asyncio.Task[NeMoGymResponse]] = field(default=None, repr=False)

    @property
    def release_file(self) -> str:
        return f"{self.root}/release"


class _Parking:
    """The activation's park handle; with ``finish_on_park`` the harness finishes before its pause is asked."""

    def __init__(self, activation: Activation, state: FakeHarnessSession, *, finish_on_park: bool) -> None:
        self._activation = activation
        self._state = state
        self._finish_on_park = finish_on_park

    @property
    def park_requested(self) -> bool:
        return self._activation.park_requested

    async def wait_park_requested(self) -> None:
        await self._activation.wait_park_requested()
        if self._finish_on_park:
            session = self._state.session
            await session.sandbox.exec(f"touch {quote(self._state.release_file)}", timeout_s=10)
            receipt = f"{session.session_dir}/{CLEANUP_RECEIPT_FILE}"
            for _ in range(600):
                if (await session.sandbox.exec(f"[ -f {quote(receipt)} ]", timeout_s=10)).return_code == 0:
                    break
                await asyncio.sleep(0.05)

    async def boundary(self, snapshot: Any) -> None:
        await self._activation.boundary(snapshot)

    def queued(self) -> Any:
        return self._activation.queued()


class FakeHarnessAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: FakeHarnessAgentConfig
    checkpoint_mode: ClassVar[str] = "exported"
    rebinds_tool_accesses: ClassVar[bool] = True
    _failed_pauses: int = 0

    async def run(self, body: BaseRunRequest) -> BaseVerifyResponse:
        raise NotImplementedError("the fake harness agent runs Environment Server sessions only")

    def _seed_restarts(self, body: AgentSeedSessionRequest) -> bool:
        return body.sandbox_access is None

    async def _connect(self, body: AgentSeedSessionRequest) -> AsyncSandbox:
        access = body.sandbox_access
        if access is None or not isinstance(access.connection, DirectSandboxConnection):
            raise AgentSeedRejectedError("the fake harness runs only in a sandbox its resources server lends")
        provider = create_provider(
            resolve_provider_config(access.connection.provider_config_ref, get_global_config_dict())
        )
        return await AsyncSandbox.connect(access.connection.descriptor, provider=provider)

    def _session(self, sandbox: AsyncSandbox, root: str, workdir: str) -> SandboxSession[None]:
        return SandboxSession(sandbox=sandbox, session_dir=root, workdir=workdir, harness="fake", pausable=True)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> FakeHarnessSession:
        sandbox = await self._connect(body)
        workdir = body.sandbox_access.workdir
        root = str(PurePosixPath(workdir).parent / "harness" / uuid4().hex)
        await self._install(sandbox, root)
        return FakeHarnessSession(
            request=body, session=self._session(sandbox, root, workdir), root=root, workdir=workdir
        )

    async def _install(self, sandbox: AsyncSandbox, root: str) -> None:
        result = await sandbox.exec(f"mkdir -p {quote(root)}", timeout_s=30)
        if result.return_code != 0:
            raise RuntimeError(f"cannot prepare the harness: {result.stderr}")
        script = Path(f"/tmp/fake-harness-{uuid4().hex}.py")
        script.write_text(HARNESS)
        try:
            await sandbox.upload(script, f"{root}/harness.py")
        finally:
            script.unlink(missing_ok=True)

    async def responses(self, request: Request, body: Any = None) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "the fake harness agent needs a seeded session")
        state = self._require_agent_session(session_id)
        assert isinstance(state, FakeHarnessSession)
        rollout_id = request.path_params["rollout_id"]
        participant = self.checkpoint_participant
        participant.check_session(session_id, EpisodeId.from_capture_key(rollout_id))
        index = self._activation_index(request)
        if index is not None and index == state.completed_activation and state.completed_reply is not None:
            participant.answered_from_saved_reply(session_id, EpisodeId.from_capture_key(rollout_id))
            return NeMoGymResponse.model_validate(state.completed_reply)
        if state.task is None:
            if state.session is None:
                raise HTTPException(409, "restored session was not re-bound")
            state.task = asyncio.create_task(self._activate(state, index, rollout_id))
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def _activate(self, state: FakeHarnessSession, index: Optional[int], rollout_id: str) -> NeMoGymResponse:
        participant = self.checkpoint_participant
        key = state.request.agent_session_id
        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        async with participant.activation(key, state.request.episode_id, index=index) as activation:
            while True:
                if state.launch_count > 0:
                    directory = f"{state.root}/launch-{state.launch_count + 1}"
                    await state.session.begin_activation(session_dir=directory, timeout=self.config.close_timeout_s)
                    await state.session.sandbox.exec(f"mkdir -p {quote(directory)}", timeout_s=30)
                state.launch_count += 1
                if self._failed_pauses < self.config.fail_pauses:
                    self._break_next_pause(state.session)
                launch = state.launch_count

                async def stage() -> SandboxCommand:
                    return SandboxCommand(
                        argv=[
                            sys.executable,
                            f"{state.root}/harness.py",
                            state.workdir,
                            str(self.config.steps),
                            str(self.config.step_delay_s),
                            base_url,
                            state.release_file if self.config.finish_on_park else "",
                        ]
                    )

                async def collect() -> None:
                    return None

                outcome = await run_until_parked(
                    state.session,
                    _Parking(activation, state, finish_on_park=self.config.finish_on_park),
                    launch=lambda: state.session.execute(
                        stage_activation=stage,
                        collect=collect,
                        timeout=600,
                        close_timeout=self.config.close_timeout_s,
                        pause_allowance_s=600,
                    ),
                    on_parked=lambda: {"launch": launch},
                    stop_timeout=self.config.close_timeout_s,
                )
                if isinstance(outcome, Finished):
                    break
            response = NeMoGymResponse(
                id=f"resp_{uuid4().hex}",
                created_at=int(time()),
                model=self.config.model_server.name,
                object="response",
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "id": f"msg_{uuid4().hex}",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ],
                tool_choice="auto",
                parallel_tool_calls=False,
                tools=[],
                metadata={"harness_launches": str(state.launch_count)},
            )
            state.observations = AgentObservationBundle(
                source="fake_harness",
                records=[AgentInvocation(invocation_id=f"{key}:activation-1", status="completed")],
            ).model_dump(mode="json")
            state.completed_activation = index if index is not None else state.completed_activation + 1
            state.completed_reply = response.model_dump(mode="json")
            return response

    def _break_next_pause(self, session: SandboxSession[None]) -> None:
        async def unconfirmed(**kwargs: Any) -> HarnessPauseResult:
            del session.pause_harness
            self._failed_pauses += 1
            return HarnessPauseResult(status="unconfirmed", error="injected: the pause was not confirmed")

        session.pause_harness = unconfirmed

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, FakeHarnessSession)
        if state.session is not None:
            await state.session.close(timeout=self.config.close_timeout_s)
        observations = (
            AgentObservationBundle.model_validate(state.observations)
            if state.observations
            else AgentObservationBundle(
                source="fake_harness", gaps=[ObservationGap(code="agent_activation_interrupted")]
            )
        )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=observations
        )

    # -- partial-rollout checkpoints ------------------------------------------------------------------------------

    async def export_agent_sessions(
        self, session_keys: list[str], *, checkpoint_id: str
    ) -> dict[str, dict[str, JsonValue]]:
        exported = {}
        for key in session_keys:
            state = self._session_state(key)
            assert isinstance(state, FakeHarnessSession)
            if state.session is None:
                exported[key] = dict(state.restored or {})
                continue
            exported[key] = {
                "request": state.request.model_dump(mode="json"),
                "session": dict(state.session.checkpoint_state()),
                "root": state.root,
                "workdir": state.workdir,
                "launch_count": state.launch_count,
                "completed_activation": state.completed_activation,
                "completed_reply": state.completed_reply,
                "observations": state.observations,
            }
        return exported

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        for restored in sessions:
            saved = restored.session
            request = AgentSeedSessionRequest.model_validate(
                saved["request"] | {"episode_id": restored.episode_id.model_dump(mode="json")}
            )
            self._install_restored_session(
                restored.session_key,
                FakeHarnessSession(
                    request=request,
                    session=None,
                    root=str(saved["root"]),
                    workdir=str(saved["workdir"]),
                    launch_count=int(saved["launch_count"]),
                    completed_activation=int(saved["completed_activation"]),
                    completed_reply=saved.get("completed_reply"),
                    observations=saved.get("observations"),
                    restored=dict(saved),
                ),
            )

    async def _rebind_restored_agent_session_state(
        self, state: AgentSessionState, body: AgentSeedSessionRequest
    ) -> AgentSessionState:
        assert isinstance(state, FakeHarnessSession)
        if state.session is not None:
            return state
        sandbox = await self._connect(body)
        state.session = SandboxSession.reattach(
            sandbox, state.restored["session"], workdir=body.sandbox_access.workdir, harness="fake", pausable=True
        )
        state.restored = None
        return state


if __name__ == "__main__":
    FakeHarnessAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = FakeHarnessAgent.run_webserver()  # noqa: F401
