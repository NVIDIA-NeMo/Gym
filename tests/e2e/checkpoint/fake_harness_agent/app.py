# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A legacy ``/run`` agent shaped like the OpenCode agent, for the checkpoint e2e suite.

It seeds a session on its resources server, which hands it the session's sandbox, and runs a long harness
command there (``run notes <n>`` in the fake sandbox backend) as one ``wait`` step. When a checkpoint asks the
run to park, it interrupts the harness, records a boundary that continues the step, and relaunches the same
command after the resume, or in the replacement attempt after a crash; the harness picks up where the file
left off, as ``opencode run --continue`` does. Then it verifies the file's contents with the resources server.
"""

import asyncio
import sys
from contextlib import AbstractAsyncContextManager, nullcontext
from pathlib import Path
from time import time
from typing import Any, ClassVar, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, JsonValue


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sandbox_notes_server"))
from fake_sandbox_provider import RemoteFakeSandboxProvider  # noqa: E402

from nemo_gym._checkpoint.agent import LegacyRun, RestoredAgentSession, require_rollout  # noqa: E402
from nemo_gym._checkpoint.steps import StepMode, seed_verify_mode  # noqa: E402
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse  # noqa: E402
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent  # noqa: E402
from nemo_gym.config_types import ResourcesServerRef  # noqa: E402
from nemo_gym.episode_types import EpisodeId  # noqa: E402
from nemo_gym.openai_utils import (  # noqa: E402
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.sandbox.api import AsyncSandbox  # noqa: E402
from nemo_gym.server_utils import get_response_json, raise_for_status  # noqa: E402


class FakeHarnessAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    sandbox_backend_url: str
    # The harness appends this many lines, one every interval_s seconds.
    lines: int = 6
    interval_s: float = 0.5


class FakeHarnessRunRequest(BaseRunRequest):
    # Rollout collection adds routing fields such as agent_ref and the attempt index.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    expected_notes: list[str]


class FakeHarnessVerifyResponse(BaseVerifyResponse):
    harness_launches: int
    harness_interrupted: bool


class _Interrupted(Exception):
    pass


class FakeHarnessAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    checkpoint_sessions_supported: ClassVar[bool] = True
    config: FakeHarnessAgentConfig

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        return {session_key: {} for session_key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        pass

    async def retire_agent_session(self, session_key: str) -> None:
        pass

    async def responses(self, request: Request, body: Any) -> NeMoGymResponse:
        raise NotImplementedError("the fake harness agent runs legacy /run episodes only")

    def _provider(self) -> RemoteFakeSandboxProvider:
        return RemoteFakeSandboxProvider(self.config.sandbox_backend_url)

    async def run(self, request: Request, body: FakeHarnessRunRequest) -> FakeHarnessVerifyResponse:
        participant = self.checkpoint_participant
        if participant is None:
            return await self._run(request, body, legacy_run=None)
        episode_id = EpisodeId.from_capture_key(require_rollout(self.rollout_id_from_run(body)))
        async with participant.legacy_run(f"run:{episode_id.rollout_id}", episode_id) as legacy_run:
            return await self._run(request, body, legacy_run=legacy_run)

    async def _run_harness(self, sandbox: AsyncSandbox, legacy_run: Optional[LegacyRun]) -> None:
        command = f"run notes {self.config.lines} {self.config.interval_s}"
        if legacy_run is None:
            await sandbox.exec(command)
            return
        exec_task = asyncio.ensure_future(sandbox.exec(command))
        park_task = asyncio.ensure_future(legacy_run.park_requested().wait())
        try:
            await asyncio.wait({exec_task, park_task}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            exec_task.cancel()
            raise
        finally:
            park_task.cancel()
        if exec_task.done():
            exec_task.result()
            return
        await sandbox.exec("interrupt notes")
        await asyncio.wait({exec_task}, timeout=30)
        if not exec_task.done():
            exec_task.cancel()
        raise _Interrupted()

    async def _run(
        self, request: Request, body: FakeHarnessRunRequest, *, legacy_run: Optional[LegacyRun]
    ) -> FakeHarnessVerifyResponse:
        continuation = (legacy_run.continuation if legacy_run is not None else None) or {}
        stage = continuation.get("next", "seed")
        cookies: dict[str, str] = dict(continuation.get("cookies", dict(request.cookies)))
        verify_mode: StepMode = continuation.get("verify_mode", "wait")
        sandbox_handle: Optional[str] = continuation.get("sandbox_handle")
        launches = int(continuation.get("launches", 0))
        interrupted = bool(continuation.get("interrupted", False))
        notes: str = str(continuation.get("notes", ""))

        async def boundary(state: dict[str, Any]) -> None:
            if legacy_run is not None:
                await legacy_run.boundary(state)

        def step(mode: StepMode) -> AbstractAsyncContextManager[None]:
            return legacy_run.step(mode) if legacy_run is not None else nullcontext()

        if stage == "seed":
            await boundary({"next": "seed"})
            async with step("replay"):
                seeded = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(seeded)
                cookies = {**cookies, **{k: str(getattr(v, "value", v)) for k, v in seeded.cookies.items()}}
                verify_mode = seed_verify_mode(seeded.headers)
                sandbox_handle = (await seeded.json())["sandbox_handle"]
            stage = "harness"

        while stage == "harness":
            await boundary(
                {
                    "next": "harness",
                    "cookies": cookies,
                    "verify_mode": verify_mode,
                    "sandbox_handle": sandbox_handle,
                    "launches": launches,
                    "interrupted": interrupted,
                }
            )
            # The resources server owns the sandbox; asking for access resumes it if a checkpoint paused it.
            access = await self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/sandbox_access", cookies=cookies
            )
            await raise_for_status(access)
            sandbox_handle = (await access.json())["connection"]["descriptor"]["sandbox_id"]
            sandbox = await AsyncSandbox.connect({"sandbox_id": sandbox_handle}, provider=self._provider())
            launches += 1
            async with step("wait"):
                try:
                    await self._run_harness(sandbox, legacy_run)
                except _Interrupted:
                    interrupted = True
                    continue
            notes = (await sandbox.exec("read notes")).stdout or ""
            stage = "verify"

        if stage == "return":
            result = dict(continuation["result"])
        else:
            await boundary(
                {
                    "next": "verify",
                    "cookies": cookies,
                    "verify_mode": verify_mode,
                    "notes": notes,
                    "launches": launches,
                    "interrupted": interrupted,
                }
            )
            response = NeMoGymResponse(
                id=f"resp_{uuid4().hex}",
                created_at=int(time()),
                model="fake-harness",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(annotations=[], text=notes, type="output_text")],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            )
            async with step(verify_mode):
                verified = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=body.model_dump() | {"response": response.model_dump(mode="json")},
                    cookies=cookies,
                )
                await raise_for_status(verified)
                result = await get_response_json(verified)
            await boundary({"next": "return", "result": result, "launches": launches, "interrupted": interrupted})

        return FakeHarnessVerifyResponse.model_validate(
            result | {"harness_launches": launches, "harness_interrupted": interrupted}
        )


if __name__ == "__main__":
    FakeHarnessAgent.run_webserver()
