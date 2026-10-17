# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A checkpointed proof refinement /run continues in a fresh agent without repeating completed turns.

The agent runs behind its real ASGI app and control routes.
Its self-call to /v1/responses also goes through the app,
so the proxy's handling of the policy model's refusals is exercised;
only the policy model and the Lean resources server are faked.
"""

import asyncio
import json
import time
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import httpx
import pytest
from omegaconf import DictConfig
from pydantic import BaseModel

from nemo_gym.server_utils import ServerClient
from responses_api_agents.proof_refinement_agent.app import (
    ProofRefinementAgent,
    ProofRefinementAgentConfig,
    ProofRefinementVerifyResponse,
)


AUTH = {"authorization": "Bearer t"}
ROLLOUT = "proof-rollout"
RUN_BODY = {
    "_ng_rollout_id": ROLLOUT,
    "responses_create_params": {
        "input": [{"role": "user", "content": "prompt-0"}],
        "model": "proof-model",
        "temperature": 0.25,
        "top_p": 0.8,
    },
    "verifier_metadata": {"theorem": "example"},
}


def _control(checkpoint_id: str = "c1", **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


def _model_response(turn: int) -> dict:
    return {
        "id": f"response-{turn}",
        "created_at": 1,
        "model": "proof-model",
        "object": "response",
        "output": [
            {
                "id": f"message-{turn}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": f"proof-{turn}", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class _Reply:
    """The parts of an aiohttp response the agent reads."""

    def __init__(
        self, status: int, body: Any, *, cookies: Optional[dict] = None, headers: Optional[dict] = None
    ) -> None:
        self.status = status
        self.ok = status < 400
        self._body = json.dumps(body).encode()
        self.cookies = SimpleCookie(cookies or {})
        self.headers = headers or {}

    async def read(self) -> bytes:
        return self._body

    async def json(self) -> Any:
        return json.loads(self._body)


class _Harness:
    """One agent process: its app, plus a fake policy model and Lean server that can pause at a step."""

    def __init__(
        self, *, checkpoint: bool = True, verify_mode: Optional[str] = "replay", stop: Optional[tuple] = None
    ) -> None:
        self.verify_mode = verify_mode
        self.stop = stop
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        # Cleared while the policy model is closed for a checkpoint: a new call waits for resume.
        self.model_open = asyncio.Event()
        self.model_open.set()
        self.hold = False
        self.waiting = 0
        self.calls: list[str] = []
        self.generated: list[int] = []
        self.verified: list[int] = []
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = DictConfig(
            {"checkpoint": {"enabled": True, "control_auth_token": "t"}} if checkpoint else {}
        )
        server_client.post = self.post
        self.agent = ProofRefinementAgent(
            config=ProofRefinementAgentConfig(
                name="proof-agent",
                host="",
                port=0,
                entrypoint="",
                resources_server={"type": "resources_servers", "name": "lean"},
                model_server={"type": "responses_api_models", "name": "policy"},
                max_correction_turns=2,
            ),
            server_client=server_client,
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.agent.setup_webserver()), base_url="http://agent"
        )

    async def pause(self, stage: str, turn: int) -> None:
        if self.stop == (stage, turn):
            self.entered.set()
            await self.release.wait()

    async def post(self, *, server_name: str, url_path: str, json: Any, cookies: Any = None) -> _Reply:
        self.calls.append(url_path)
        if self.hold and server_name != "proof-agent":
            await asyncio.Event().wait()
        if isinstance(json, BaseModel):
            json = json.model_dump(exclude_unset=True)
        if server_name == "proof-agent":
            # The agent's self-call, through its own /v1/responses proxy.
            cookie_header = "; ".join(f"{k}={getattr(v, 'value', v)}" for k, v in (cookies or {}).items())
            reply = await self.client.post(url_path, json=json, headers={"cookie": cookie_header})
            return _Reply(reply.status_code, reply.json(), cookies=dict(reply.cookies))
        if server_name == "policy":
            if not self.model_open.is_set():
                self.waiting += 1
                await self.model_open.wait()
            turn = int(json["input"][0]["content"].rsplit("-", 1)[1])
            await self.pause("model", turn)
            self.generated.append(turn)
            return _Reply(200, _model_response(turn), cookies={"model": f"turn-{turn}"})
        if url_path == "/seed_session":
            headers = {"x-ng-checkpoint-verify": self.verify_mode} if self.verify_mode else {}
            return _Reply(200, {}, cookies={"session": "seeded"}, headers=headers)
        assert (server_name, url_path) == ("lean", "/verify")
        turn = json["turn_index"]
        await self.pause("verify", turn)
        self.verified.append(turn)
        success = turn == 2
        return _Reply(
            200,
            {
                "responses_create_params": json["responses_create_params"],
                "response": json["response"],
                "reward": float(success),
                "proof_status": "completed" if success else "failed",
                "needs_correction": not success,
                "error_feedback": None if success else f"error-{turn}",
                "correction_prompt": None if success else f"prompt-{turn + 1}",
                # The agent's own signed session cookie differs per process, so only the fakes' cookies count.
                "cookies_seen": {k: v for k, v in (cookies or {}).items() if k in ("session", "model", "verified")},
            },
            cookies={"verified": f"turn-{turn}"},
        )

    async def run(self, attempt: int = 0) -> httpx.Response:
        return await self.client.post("/run", json=RUN_BODY | {"_ng_attempt_index": attempt})

    async def control(self, operation: str, **extra: Any) -> dict:
        reply = await self.client.post(f"/ng-control/v1/checkpoint/{operation}", json=_control(**extra), headers=AUTH)
        assert reply.status_code == 200, reply.text
        return reply.json()


async def _baseline() -> dict:
    harness = _Harness(checkpoint=False)
    async with harness.client:
        reply = await harness.run()
    assert harness.generated == harness.verified == [0, 1, 2]
    return reply.json()


async def _checkpoint(harness: _Harness, directory: Path, *, wait_for_step: bool) -> tuple[dict, asyncio.Task]:
    """Prepare and commit while the run is paused; ``wait_for_step`` releases a step the checkpoint waits on."""
    task = asyncio.create_task(harness.run())
    await asyncio.wait_for(harness.entered.wait(), 2)
    prepare = asyncio.create_task(harness.control("prepare"))
    if wait_for_step:
        await asyncio.sleep(0.05)
        assert not prepare.done()
        harness.release.set()
    prepared = await prepare
    assert prepared["phase"] == "prepared", prepared
    committed = await harness.control("commit", checkpoint_dir=str(directory))
    return committed, task


@pytest.mark.parametrize(
    ("stop", "verify_mode", "next_step", "turn", "generated", "verified"),
    [
        # Generation in flight: the boundary before it is exported and the generation runs again.
        (("model", 0), "replay", "generate", 0, [0, 1, 2], [0, 1, 2]),
        # A replayable verification in flight: the generated proof is kept and only verified again.
        (("verify", 0), "replay", "verify", 0, [1, 2], [0, 1, 2]),
        # A verification the checkpoint waits on: the run parks before the correction generation.
        (("verify", 0), None, "generate", 1, [1, 2], [1, 2]),
        (("model", 1), "replay", "generate", 1, [1, 2], [1, 2]),
        # The final verification finishes during the checkpoint: the run parks with its result.
        (("verify", 2), None, "return", 2, [], []),
    ],
)
async def test_restored_run_continues_from_its_boundary_in_a_fresh_agent(
    tmp_path: Path,
    stop: tuple,
    verify_mode: Optional[str],
    next_step: str,
    turn: int,
    generated: list[int],
    verified: list[int],
) -> None:
    expected = await _baseline()

    original = _Harness(verify_mode=verify_mode, stop=stop)
    async with original.client:
        committed, task = await _checkpoint(original, tmp_path, wait_for_step=verify_mode is None)
        # The original process stops making progress, as if it crashed, and the controller retires its attempt.
        original.hold = True
        await original.control("resume")
        await original.control("retire", episode_ids=[{"rollout_id": ROLLOUT}])
        await asyncio.gather(task, return_exceptions=True)
        # The retired attempt is gone: a later checkpoint has nothing of it to export.
        after_retire = await original.control("prepare", checkpoint_id="c2")
        await original.control("resume", checkpoint_id="c2")
    assert committed["episode_ids"] == [ROLLOUT]
    assert after_retire["report"]["counts"]["sessions"] == 0
    [records] = (tmp_path / "gym" / "agent" / "proof-agent").glob("records-*.jsonl")
    [record] = map(json.loads, records.read_text().splitlines())
    assert record["episode"]["next"] == next_step
    assert record["episode"]["turn_index"] == turn
    # Completed attempts travel in the boundary, so the replacement never redoes them.
    assert len(record["episode"]["all_attempts"]) == turn + (next_step == "return")

    replacement = _Harness(verify_mode=verify_mode)
    async with replacement.client:
        await replacement.control(
            "restore", checkpoint_id="r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": ROLLOUT}]
        )
        await replacement.control("resume", checkpoint_id="r1")
        reply = await replacement.run(attempt=1)

    assert reply.status_code == 200, reply.text
    assert reply.json() == expected
    assert replacement.generated == generated
    assert replacement.verified == verified
    assert "/seed_session" not in replacement.calls
    result = ProofRefinementVerifyResponse.model_validate(reply.json())
    assert result.reward == 1.0 and result.total_turns == 3
    assert [attempt["generation"] for attempt in result.all_attempts] == ["proof-0", "proof-1", "proof-2"]


async def test_a_policy_call_waiting_for_resume_does_not_block_prepare() -> None:
    harness = _Harness()
    async with harness.client:
        # The policy model closed first, as it does in every prepare; this agent is still open.
        harness.model_open.clear()
        task = asyncio.create_task(harness.run())
        async with asyncio.timeout(2):
            while harness.waiting == 0:
                await asyncio.sleep(0.01)
        # The call is in its replay step, which a restored agent sends again, so it does not hold up prepare.
        prepared = await harness.control("prepare")
        harness.model_open.set()
        await harness.control("resume")
        reply = await asyncio.wait_for(task, 2)

    assert prepared["phase"] == "prepared"
    assert reply.status_code == 200, reply.text
    assert reply.json() == await _baseline()
    assert harness.generated == [0, 1, 2]


async def test_a_run_without_a_rollout_id_is_refused_when_checkpointing() -> None:
    harness = _Harness()
    async with harness.client:
        body = {key: value for key, value in RUN_BODY.items() if key != "_ng_rollout_id"}
        reply = await harness.client.post("/run", json=body)

    assert reply.status_code == 400
    assert reply.json()["error"]["code"] == "rollout_id_required"
    assert harness.calls == []
