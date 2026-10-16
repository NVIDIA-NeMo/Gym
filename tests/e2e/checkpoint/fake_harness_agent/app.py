# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A legacy ``/run`` agent shaped like the OpenCode agent, for the checkpoint e2e suite.

It runs on ``SandboxedHarnessAgent`` with a fake adapter, so the suite's process-level scenarios exercise the
driver OpenCode runs on. The resources server owns the session's sandbox and names it in the seed reply; the
harness is a long command there (``run notes <n>`` in the fake sandbox backend). When a checkpoint asks the run to
park, the adapter interrupts the harness and the driver records a bookmark that continues it; the relaunch, after
the resume or in the replacement attempt after a crash, picks up where the file left off, as
``opencode run --continue`` does. Then the adapter verifies the file's contents with the resources server.

Before each launch the driver asks the resources server's ``/sandbox_access`` which sandbox is current, because a
restore forks the sandbox under a new ID. ``refresh_sandbox_access: false`` launches in the sandbox the bookmark
names instead, which after a restore is the superseded one: the negative control.
"""

import asyncio
import sys
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict, JsonValue


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sandbox_notes_server"))
from fake_sandbox_provider import RemoteFakeSandboxProvider  # noqa: E402

from nemo_gym._checkpoint.harness import HarnessInterrupted, ResumeDecision  # noqa: E402
from nemo_gym._checkpoint.harness_run import LaunchContext, LentSandbox, SandboxedHarnessAgent  # noqa: E402
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse  # noqa: E402
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig  # noqa: E402
from nemo_gym.config_types import ResourcesServerRef  # noqa: E402
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
    refresh_sandbox_access: bool = True


class FakeHarnessRunRequest(BaseRunRequest):
    # Rollout collection adds routing fields such as agent_ref and the attempt index.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    expected_notes: list[str]


class FakeHarnessVerifyResponse(BaseVerifyResponse):
    harness_launches: int
    harness_interrupted: bool


class FakeHarnessOutcome(BaseModel):
    notes: str


class FakeHarnessAgent(SandboxedHarnessAgent):
    ray_enabled = False
    config: FakeHarnessAgentConfig

    async def responses(self, request: Request, body: Any) -> NeMoGymResponse:
        raise NotImplementedError("the fake harness agent runs legacy /run episodes only")

    def provider(self) -> RemoteFakeSandboxProvider:
        return RemoteFakeSandboxProvider(self.config.sandbox_backend_url)

    async def run(self, request: Request, body: FakeHarnessRunRequest) -> FakeHarnessVerifyResponse:
        done = await self.run_harness(request, body, FakeHarnessAdapter(self, body))
        return FakeHarnessVerifyResponse.model_validate(
            done.result
            | {
                "harness_launches": len(done.launches),
                "harness_interrupted": any(launch.ended == "interrupted" for launch in done.launches),
            }
        )

    async def refresh_lent_sandbox(
        self, cookies: dict[str, str], hint: Optional[LentSandbox]
    ) -> Optional[LentSandbox]:
        if not self.config.refresh_sandbox_access:
            return hint
        return await super().refresh_lent_sandbox(cookies, hint)


class FakeHarnessAdapter:
    """The notes harness: its progress is the notes file in the sandbox, so every relaunch continues it."""

    outcome_model = FakeHarnessOutcome

    def __init__(self, agent: FakeHarnessAgent, body: FakeHarnessRunRequest) -> None:
        self._agent = agent
        self._body = body

    def new_state(self) -> dict[str, JsonValue]:
        return {}

    async def connect(self, lent: Optional[LentSandbox]) -> AsyncSandbox:
        if lent is None or lent.sandbox_id is None:
            raise RuntimeError("the fake harness runs only in a sandbox its resources server lends")
        return await AsyncSandbox.connect({"sandbox_id": lent.sandbox_id}, provider=self._agent.provider())

    async def resume_decision(self, sandbox: AsyncSandbox, state: dict[str, JsonValue]) -> ResumeDecision:
        return ResumeDecision(resume=True, reason="the notes file carries the harness's progress")

    async def launch(self, launch: LaunchContext) -> FakeHarnessOutcome:
        # The fake backend has no processes to signal: the harness is interrupted through the backend instead of
        # launch.run's tagged stop.
        sandbox = launch.sandbox
        exec_task = asyncio.ensure_future(
            sandbox.exec(f"run notes {self._agent.config.lines} {self._agent.config.interval_s}")
        )
        park_task = asyncio.ensure_future(launch.wait_park_requested())
        try:
            await asyncio.wait({exec_task, park_task}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            exec_task.cancel()
            raise
        finally:
            park_task.cancel()
        if not exec_task.done():
            await sandbox.exec("interrupt notes")
            await asyncio.wait({exec_task}, timeout=30)
            if not exec_task.done():
                exec_task.cancel()
            raise HarnessInterrupted("interrupted the notes harness")
        exec_task.result()
        return FakeHarnessOutcome(notes=(await sandbox.exec("read notes")).stdout or "")

    async def verify(self, cookies: dict[str, str], outcome: FakeHarnessOutcome) -> dict[str, Any]:
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model="fake-harness",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(annotations=[], text=outcome.notes, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )
        verified = await self._agent.server_client.post(
            server_name=self._agent.config.resources_server.name,
            url_path="/verify",
            json=self._body.model_dump() | {"response": response.model_dump(mode="json")},
            cookies=cookies,
        )
        await raise_for_status(verified)
        return await get_response_json(verified)


if __name__ == "__main__":
    FakeHarnessAgent.run_webserver()
