# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual agent and checkpoint lifecycle with a controlled model and real verifier."""

import asyncio
import json
import re
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime import ActivationOutOfOrderError
from nemo_gym.server_runtime._checkpoint.agent import AgentSessionRecord
from nemo_gym.server_runtime._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    RestoreRequest,
)
from nemo_gym.server_runtime._checkpoint.errors import StaleAttemptError
from resources_servers.single_step_tool_use_with_argument_comparison.app import (
    SingleStepToolUseArgumentComparisonResourcesServer,
    SingleStepToolUseArgumentComparisonResourcesServerConfig,
    SingleStepToolUseArgumentComparisonVerifyRequest,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ToolCallComparatorConfig,
)
from responses_api_agents.tool_simulation_agent.app import (
    ToolSimulationAgent,
    ToolSimulationAgentConfig,
    ToolSimulationAgentRunRequest,
)


MODEL_REPLY = {
    "id": "model-call",
    "object": "response",
    "created_at": 1,
    "model": "model",
    "output": [{"type": "function_call", "name": "acknowledge", "arguments": "{}", "call_id": "tool-call"}],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def row(attempt: int = 0) -> ToolSimulationAgentRunRequest:
    return ToolSimulationAgentRunRequest.model_validate(
        {
            "_ng_rollout_id": EpisodeId(rollout_id="rollout", attempt=attempt).capture_key,
            "responses_create_params": {"input": "Call acknowledge."},
            "expected_action": {"type": "function_call", "name": "acknowledge", "arguments": "{}"},
        }
    )


def control(checkpoint_id: str = "c1") -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5}


def request(capture_key: str) -> Request:
    return Request(
        {
            "type": "http",
            "path": f"/ng-rollout/{capture_key}/v1/responses",
            "query_string": b"",
            "path_params": {"rollout_id": capture_key},
            "headers": [],
        }
    )


class Reply:
    ok = True

    def __init__(self, payload: dict, *, headers: dict | None = None, cookies: dict | None = None):
        self.payload, self.headers, self.cookies = payload, headers or {}, cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    async def json(self) -> dict:
        return self.payload


class Harness:
    def __init__(
        self,
        *,
        skip_verification: bool = False,
        restart_resource: bool = False,
        fail_verification: bool = False,
        detached_dispatch: bool = False,
    ):
        self.model_started, self.model_release = asyncio.Event(), asyncio.Event()
        self.verify_started, self.verify_release = asyncio.Event(), asyncio.Event()
        self.reply_sent, self.reply_delivery = asyncio.Event(), asyncio.Event()
        self.verify_release.set()
        self.reply_delivery.set()
        self.model_paths: list[str] = []
        self.verifications = self.closes = 0
        self.restart_resource = restart_resource
        self.fail_verification = fail_verification
        self.detached_dispatch = detached_dispatch
        self.model_handlers: list[asyncio.Task] = []
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
        client.post = AsyncMock(side_effect=self.post)
        self.agent = ToolSimulationAgent(
            config=ToolSimulationAgentConfig(
                host="127.0.0.1",
                port=0,
                entrypoint="app.py",
                name="agent",
                resources_server={"type": "resources_servers", "name": "verifier"},
                model_server={"type": "responses_api_models", "name": "model"},
                skip_verification=skip_verification,
                skip_verification_reward=0.75,
            ),
            server_client=client,
        )
        self.agent.setup_webserver()
        self.participant = self.agent.checkpoint_participant
        assert self.participant is not None
        self.controller = ParticipantControlPlane(self.participant, instance_name="agent", lease_grace_seconds=60)
        self.verifier = SingleStepToolUseArgumentComparisonResourcesServer(
            config=SingleStepToolUseArgumentComparisonResourcesServerConfig(
                host="127.0.0.1",
                port=0,
                entrypoint="app.py",
                name="verifier",
                tool_call_comparator_config=ToolCallComparatorConfig(word_count_similarity_threshold=0.1),
            ),
            server_client=client,
        )
        self.tasks: list[asyncio.Task] = []

    async def post(self, *, server_name, url_path, json, **kwargs):
        if server_name == "model":
            self.model_paths.append(url_path)
            self.model_started.set()
            await self.model_release.wait()
            return Reply(MODEL_REPLY)
        if server_name == "agent":
            capture_key = re.search(r"/ng-rollout/([^/]+)", url_path)[1]
            if self.detached_dispatch:
                # Cancelling an HTTP caller need not cancel the server's handler.
                handler = asyncio.create_task(self.agent.responses(request(capture_key), json))
                self.model_handlers.append(handler)
                self.tasks.append(handler)
                response = await asyncio.shield(handler)
            else:
                response = await self.agent.responses(request(capture_key), json)
            self.reply_sent.set()
            await self.reply_delivery.wait()
            return Reply(response.model_dump(mode="json"))
        if url_path == "/seed_session":
            headers = {"x-ng-checkpoint-verify": "replay"}
            if self.restart_resource:
                headers["x-ng-checkpoint-restart"] = "1"
            return Reply({}, headers=headers, cookies={"session": "resource-session"})
        if url_path == "/verify":
            self.verifications += 1
            self.verify_started.set()
            await self.verify_release.wait()
            if self.fail_verification:
                raise RuntimeError("controlled verification failure")
            verdict = await self.verifier.verify(SingleStepToolUseArgumentComparisonVerifyRequest.model_validate(json))
            return Reply(verdict.model_dump(mode="json"))
        if url_path == "/close_session":
            self.closes += 1
            assert kwargs["cookies"] == {"session": "resource-session"}
            return Reply({})
        raise AssertionError((server_name, url_path))

    def start(self, attempt: int = 0) -> asyncio.Task:
        task = asyncio.create_task(self.agent.run(row(attempt)))
        self.tasks.append(task)
        return task

    async def prepare(self, checkpoint_id="c1") -> dict:
        result = await self.controller.prepare(CheckpointRequest(**control(checkpoint_id)))
        assert result["report"]["ready"], result
        return result

    async def save(self, directory: Path, checkpoint_id="c1") -> list[AgentSessionRecord]:
        await self.prepare(checkpoint_id)
        await self.controller.commit(CommitRequest(**control(checkpoint_id), checkpoint_dir=str(directory)))
        return await self.participant.export(None)

    async def restore(self, directory: Path, attempt=0, checkpoint_id="c1") -> None:
        await self.controller.restore(
            RestoreRequest(
                **control(checkpoint_id),
                checkpoint_dir=str(directory),
                episode_ids=[EpisodeId(rollout_id="rollout", attempt=attempt)],
            )
        )
        await self.controller.resume(CheckpointRequest(**control(checkpoint_id)))

    async def close(self) -> None:
        # Replay-step cancellation parks until resume; reopen before joining it.
        if self.controller.checkpoint_id is not None:
            await self.controller.resume(CheckpointRequest(**control(self.controller.checkpoint_id)))
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


@pytest.fixture
async def harnesses():
    values = []

    def make(**kwargs):
        value = Harness(**kwargs)
        values.append(value)
        return value

    yield make
    for value in values:
        await value.close()


async def test_inflight_model_wait_saves_and_survives_a_second_restore(harnesses, tmp_path):
    first = harnesses()
    run = first.start()
    await asyncio.wait_for(first.model_started.wait(), 2)
    records = await first.save(tmp_path / "first")
    assert len(records) == 1 and records[0].boundary == {}
    assert records[0].episode["next"] == "model"
    assert not first.participant.readiness().restarts
    first.model_release.set()
    await asyncio.sleep(0.02)
    assert not run.done(), "late model reply must stay parked until resume"

    second = harnesses()
    await second.restore(tmp_path / "first")
    second.start(1)
    await asyncio.wait_for(second.model_started.wait(), 2)
    records = await second.save(tmp_path / "second", "c2")
    assert records[0].episode_id == EpisodeId(rollout_id="rollout", attempt=1)
    assert records[0].boundary == {} and records[0].episode["next"] == "model"

    third = harnesses()
    await third.restore(tmp_path / "second", attempt=1, checkpoint_id="c2")
    third.model_release.set()
    result = await asyncio.wait_for(third.start(2), 2)
    assert result.reward == 1.0
    assert "/ng-rollout/rollout-a2/" in third.model_paths[0]
    assert not third.agent._model_replies
    assert third.closes == 1


@pytest.mark.parametrize("skip_verification", [False, True])
async def test_completed_model_reply_on_wire_is_restored_without_new_generation(
    harnesses, tmp_path, skip_verification
):
    first = harnesses(skip_verification=skip_verification)
    first.model_release.set()
    first.reply_delivery.clear()
    first.start()
    await asyncio.wait_for(first.reply_sent.wait(), 2)
    records = await first.save(tmp_path)
    assert records[0].session["response"]["id"] == MODEL_REPLY["id"]
    assert records[0].episode["next"] == "model"
    second = harnesses(skip_verification=skip_verification)
    await second.restore(tmp_path)
    result = await asyncio.wait_for(second.start(1), 2)
    assert result.reward == (0.75 if skip_verification else 1.0) and second.model_paths == []
    assert second.verifications == (0 if skip_verification else 1)


async def test_checkpoint_in_verification_reuses_response_and_replays_only_verification(harnesses, tmp_path):
    first = harnesses()
    first.model_release.set()
    first.verify_release.clear()
    first.start()
    await asyncio.wait_for(first.verify_started.wait(), 2)
    records = await first.save(tmp_path)
    assert records[0].episode["next"] == "verify"
    assert "response" not in records[0].session, "boundary owns the response without a second copy"
    with pytest.raises(ActivationOutOfOrderError):
        await first.agent.responses(request("rollout"), row().responses_create_params)
    second = harnesses()
    await second.restore(tmp_path)
    result = await asyncio.wait_for(second.start(1), 2)
    assert result.reward == 1.0 and second.model_paths == []


async def test_failed_verification_during_checkpoint_can_restart_after_restore(harnesses, tmp_path):
    first = harnesses(fail_verification=True)
    first.model_release.set()
    first.verify_release.clear()
    run = first.start()
    await asyncio.wait_for(first.verify_started.wait(), 2)
    await first.prepare()
    first.verify_release.set()
    async with asyncio.timeout(2):
        while first.participant.legacy_episodes.exported()["run:rollout"] is not None:
            await asyncio.sleep(0)
    assert not run.done(), "the failed replay step must stay parked until resume"
    await first.controller.commit(CommitRequest(**control(), checkpoint_dir=str(tmp_path)))
    records = await first.participant.export(None)
    assert records[0].episode is None
    assert records[0].session == {"model_completed": False}

    second = harnesses()
    await second.restore(tmp_path)
    second.model_release.set()
    result = await asyncio.wait_for(second.start(1), 2)
    assert result.reward == 1.0
    assert second.model_paths == ["/ng-rollout/rollout-a1/v1/responses"]
    assert second.verifications == 1
    await first.controller.resume(CheckpointRequest(**control()))
    with pytest.raises(RuntimeError, match="controlled verification failure"):
        await asyncio.wait_for(run, 2)


@pytest.mark.parametrize("replacement_live", [False, True])
async def test_cancelled_run_late_model_reply_cannot_be_reused_by_replacement(harnesses, replacement_live):
    first = harnesses(detached_dispatch=True)
    old_run = first.start()
    await asyncio.wait_for(first.model_started.wait(), 2)
    old_run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await old_run
    assert not first.participant.has_session("run:rollout")
    assert not first.model_handlers[0].done(), "the self-dispatched HTTP handler is still running"
    if replacement_live:
        first.participant.open_session("run:rollout", EpisodeId(rollout_id="rollout", attempt=1))
    first.model_release.set()
    await asyncio.wait_for(first.model_handlers[0], 2)
    result = await asyncio.wait_for(first.start(1), 2)
    assert result.reward == 1.0
    assert first.model_paths == ["/ng-rollout/rollout/v1/responses", "/ng-rollout/rollout-a1/v1/responses"]


async def test_unclaimed_restored_reply_survives_another_save(harnesses, tmp_path):
    first = harnesses()
    first.model_release.set()
    first.reply_delivery.clear()
    first.start()
    await asyncio.wait_for(first.reply_sent.wait(), 2)
    await first.save(tmp_path / "first")
    second = harnesses()
    await second.restore(tmp_path / "first")
    await second.save(tmp_path / "second", "c2")
    third = harnesses()
    await third.restore(tmp_path / "second", attempt=1, checkpoint_id="c2")
    assert (await asyncio.wait_for(third.start(2), 2)).reward == 1.0
    assert third.model_paths == []


async def test_old_attempt_cannot_consume_the_restored_reply(harnesses, tmp_path):
    first = harnesses()
    first.model_release.set()
    first.reply_delivery.clear()
    first.start()
    await asyncio.wait_for(first.reply_sent.wait(), 2)
    await first.save(tmp_path)
    second = harnesses()
    await second.restore(tmp_path)
    with pytest.raises(StaleAttemptError):
        await second.agent.responses(request("rollout"), row().responses_create_params)
    assert "run:rollout" in second.agent._model_replies
    assert (await asyncio.wait_for(second.start(1), 2)).reward == 1.0


async def test_unsupported_resource_keeps_restart_fallback(harnesses):
    first = harnesses(restart_resource=True)
    run = first.start()
    await asyncio.wait_for(first.model_started.wait(), 2)
    await first.prepare()
    assert first.participant.readiness().restarts == ["rollout"]
    assert await first.participant.export(None) == []
    first.model_release.set()
    await first.controller.resume(CheckpointRequest(**control()))
    assert (await asyncio.wait_for(run, 2)).reward == 1.0


async def test_direct_model_call_remains_restart_only(harnesses):
    first = harnesses()
    task = asyncio.create_task(first.agent.responses(request("rollout"), row().responses_create_params))
    first.tasks.append(task)
    await asyncio.wait_for(first.model_started.wait(), 2)
    await first.prepare()
    assert first.participant.readiness().restarts == ["rollout"]
    first.model_release.set()
    assert (await asyncio.wait_for(task, 2)).id == MODEL_REPLY["id"]


def test_comparison_verifier_is_stateless_and_replayable():
    assert SingleStepToolUseArgumentComparisonResourcesServer.checkpoint_mode == "stateless"
    assert SingleStepToolUseArgumentComparisonResourcesServer.checkpoint_verify == "replay"


async def test_completed_verification_result_restores_without_model_or_verifier(harnesses, tmp_path):
    first = harnesses()
    first.model_release.set()
    first.verify_release.clear()
    first.start()
    await asyncio.wait_for(first.verify_started.wait(), 2)
    await first.prepare()
    first.verify_release.set()
    async with asyncio.timeout(2):
        while first.participant.legacy_episodes.exported()["run:rollout"]["next"] != "return":
            await asyncio.sleep(0)
    await first.controller.commit(CommitRequest(**control(), checkpoint_dir=str(tmp_path)))
    second = harnesses()
    await second.restore(tmp_path)
    result = await asyncio.wait_for(second.start(1), 2)
    assert result.reward == 1.0
    assert second.model_paths == [] and second.verifications == 0


async def test_real_verifier_seed_advertises_replay_without_restart(harnesses):
    import httpx

    harness = harnesses()
    app = harness.verifier.setup_webserver()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://verifier") as client:
        response = await client.post("/ng-rollout/rollout/seed_session", json=row().model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert response.headers["x-ng-checkpoint-verify"] == "replay"
        assert response.headers.get("x-ng-checkpoint-restart") != "1"
