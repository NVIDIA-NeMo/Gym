# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise persisted proof continuation through the registered legacy run route."""

import asyncio
import json
import time
from copy import deepcopy
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from nemo_gym._checkpoint.agent import RestoredAgentSession
from nemo_gym._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantController,
    RestoreRequest,
    RetireRequest,
)
from nemo_gym._checkpoint.errors import RolloutIdRequiredError, StaleAttemptError
from nemo_gym._checkpoint.steps import CHECKPOINT_VERIFY_HEADER
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.lineage import InMemoryLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext, reset_token_sink, resolve_parent, set_token_sink
from responses_api_agents.proof_refinement_agent.app import (
    ProofRefinementAgent,
    ProofRefinementAgentConfig,
    ProofRefinementRunRequest,
)


def _reply(body, *, cookies=None, headers=None, status=200):
    reply = AsyncMock()
    reply.ok = status < 400
    reply.status = status
    reply.cookies = SimpleCookie(cookies or {})
    reply.headers = headers or {}
    reply.read.return_value = json.dumps(body).encode()
    return reply


def _model_response(turn):
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


def _request(path="/run", *, cookies=None, capture_key=None):
    return Request(
        {
            "type": "http",
            "path": path,
            "headers": [(b"cookie", "; ".join(f"{k}={v}" for k, v in (cookies or {}).items()).encode())],
            "path_params": {"rollout_id": capture_key} if capture_key is not None else {},
        }
    )


def _control(checkpoint_id="save", **kwargs):
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **kwargs}


class _Run:
    """Real agent/controller, with only model and Lean HTTP traffic scripted."""

    def __init__(
        self,
        *,
        stop=None,
        verify_mode="replay",
        enabled=True,
        success_turn=2,
        max_corrections=2,
        include_attempts=True,
        correction=True,
    ):
        self.stop = stop
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.verify_mode = verify_mode
        self.success_turn = success_turn
        self.correction = correction
        self.calls = []
        self.generated = []
        self.verified = []
        self.lineage = InMemoryLineageStore()
        self.parents = []
        self.refuse_once = None
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {"checkpoint": {"enabled": enabled, "control_auth_token": "test"}}
        client.post = AsyncMock(side_effect=self.post)
        self.agent = ProofRefinementAgent(
            config=ProofRefinementAgentConfig(
                name="proof-agent",
                host="127.0.0.1",
                port=8080,
                entrypoint="app.py",
                resources_server={"type": "resources_servers", "name": "lean"},
                model_server={"type": "responses_api_models", "name": "policy"},
                max_correction_turns=max_corrections,
                include_all_attempts=include_attempts,
            ),
            server_client=client,
        )
        app = self.agent.setup_webserver()
        self.run_endpoint = next(route.endpoint for route in app.routes if route.path == "/run")
        self.participant = self.agent.checkpoint_participant
        self.controller = (
            ParticipantController(self.participant, instance_name="proof-agent", lease_grace_seconds=60)
            if enabled
            else None
        )

    async def block(self, stage, turn):
        if self.stop == (stage, turn):
            self.entered.set()
            await self.release.wait()

    async def post(self, *, server_name, url_path, json, cookies):
        payload = json.model_dump(mode="json") if hasattr(json, "model_dump") else deepcopy(json)
        self.calls.append((server_name, url_path, payload, dict(cookies)))
        if url_path == "/seed_session":
            await self.block("seed", 0)
            headers = {CHECKPOINT_VERIFY_HEADER: self.verify_mode} if self.verify_mode else {}
            return _reply({}, cookies={"seed": "saved"}, headers=headers)
        if server_name == "policy":
            prompt = payload["input"][-1]["content"]
            turn = int(prompt.rsplit("-", 1)[-1]) if prompt.startswith("correction-") else 0
            if self.refuse_once is not None:
                refused, self.refuse_once = self.refuse_once, None
                await self.participant.close_admission(CheckpointRequest(**_control()))
                self.entered.set()
                return _reply(refused, status=409)
            await self.block("model", turn)
            capture_key = url_path.split("/")[2] if url_path.startswith("/ng-rollout/") else "plain"
            context = CaptureContext(
                rollout_id=capture_key,
                model_call_id=f"call-{turn}",
                token_sink=None,
                lineage_store=self.lineage,
                external_staging=True,
            )
            token = set_token_sink(context)
            try:
                await resolve_parent(payload["input"])
            finally:
                reset_token_sink(token)
            self.parents.append(context.capture_admission)
            self.lineage.index.for_rollout(capture_key).record(
                f"call-{turn}",
                payload["input"] + _model_response(turn)["output"],
                cum_tokens=[turn + 1],
                digest=f"proof-{turn}",
            )
            self.generated.append(turn)
            reply = _reply(_model_response(turn), cookies={"model": f"turn{turn}"})
            if self.stop == ("model_read", turn):

                async def read():
                    await self.block("model_read", turn)
                    return reply.read.return_value

                reply.read.side_effect = read
            return reply
        assert url_path == "/verify"
        turn = payload["turn_index"]
        await self.block("verify", turn)
        self.verified.append(turn)
        success = turn >= self.success_turn
        result = {
            "responses_create_params": payload["responses_create_params"],
            "response": payload["response"],
            "reward": float(success),
            "proof_status": "completed" if success else "failed",
            "needs_correction": not success,
            "error_feedback": None if success else "type mismatch",
            "correction_prompt": f"correction-{turn + 1}" if self.correction else None,
        }
        return _reply(result, cookies={"verify": f"turn{turn}"})

    async def run(self, attempt=0):
        body = ProofRefinementRunRequest.model_validate(
            {
                "responses_create_params": {"input": "prove theorem", "temperature": 0.0, "top_p": 0.8},
                "_ng_rollout_id": "proof",
                "_ng_attempt_index": attempt,
                "formal_statement": "example : True := by",
            }
        )
        return await self.run_endpoint(request=_request(cookies={"caller": "original"}), body=body)

    async def commit(self, directory: Path, checkpoint_id="save"):
        prepared = await self.controller.prepare(CheckpointRequest(**_control(checkpoint_id)))
        assert prepared["phase"] == "prepared"
        records = await self.participant.export(None)
        committed = await self.controller.commit(
            CommitRequest(**_control(checkpoint_id, checkpoint_dir=str(directory)))
        )
        assert committed["phase"] == "committed"
        assert list(directory.rglob("*.json"))
        return records

    async def restore(self, directory: Path, *, source_attempt=0):
        restored = await self.controller.restore(
            RestoreRequest(
                **_control(
                    "restore",
                    checkpoint_dir=str(directory),
                    episode_ids=[EpisodeId(rollout_id="proof", attempt=source_attempt)],
                )
            )
        )
        assert restored["restored"] == [EpisodeId(rollout_id="proof", attempt=source_attempt + 1).capture_key]
        await self.controller.resume(CheckpointRequest(**_control("restore")))


async def _cancel(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [("seed", 0), ("model", 0), ("verify", 0), ("model", 1), ("verify", 1)])
async def test_disk_restore_matches_uninterrupted_progress(tmp_path, stop):
    baseline = await _Run().run()
    source = _Run(stop=stop)
    task = asyncio.create_task(source.run())
    await asyncio.wait_for(source.entered.wait(), 5)
    [record] = await source.commit(tmp_path)
    assert record.episode["next"] == stop[0]
    assert len(record.episode["all_attempts"]) == stop[1]
    if stop[0] == "verify":
        assert record.episode["pending_response"]["id"] == f"response-{stop[1]}"
    await _cancel(task)

    restored = _Run()
    await restored.restore(tmp_path)
    result = await asyncio.wait_for(restored.run(1), 5)
    assert result.model_dump() == baseline.model_dump()
    assert source.generated + restored.generated == [0, 1, 2]
    assert source.verified + restored.verified == [0, 1, 2]
    assert all(admission.mode == "text" and admission.parent_call_id is None for admission in restored.parents)
    assert not restored.participant.has_session("run:proof")
    assert await restored.participant.export(None) == []
    if stop[0] != "seed":
        assert all(path != "/seed_session" for _, path, _, _ in restored.calls)
        assert restored.calls[0][3]["seed"] == "saved"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [("model", 1), ("verify", 1)])
async def test_second_checkpoint_before_restored_run_preserves_progress(tmp_path, stop):
    source = _Run(stop=stop)
    task = asyncio.create_task(source.run())
    await asyncio.wait_for(source.entered.wait(), 5)
    [original] = await source.commit(tmp_path / "first")
    await _cancel(task)
    intermediate = _Run()
    await intermediate.restore(tmp_path / "first")
    [saved_again] = await intermediate.commit(tmp_path / "second", "again")
    assert saved_again.episode == original.episode
    assert saved_again.episode_id.attempt == 1
    restored = _Run()
    await restored.restore(tmp_path / "second", source_attempt=1)
    result = await restored.run(2)
    assert result.model_dump() == (await _Run().run()).model_dump()
    assert restored.generated == ([1, 2] if stop[0] == "model" else [2])
    assert restored.verified == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("success_turn", [0, 2])
async def test_nonreplayable_verify_waits_and_saves_completed_result(tmp_path, success_turn):
    source = _Run(stop=("verify", 0), verify_mode=None, success_turn=success_turn)
    task = asyncio.create_task(source.run())
    await asyncio.wait_for(source.entered.wait(), 5)
    prepared = await source.controller.prepare(CheckpointRequest(**_control(deadline_ts=time.time() + 0.03)))
    assert prepared["phase"] == "preparing"
    assert prepared["report"]["blockers"] == ["proof"]
    source.release.set()
    [record] = await source.commit(tmp_path)
    assert record.episode["all_attempts"][0]["generation"] == "proof-0"
    assert record.episode["next"] == ("return" if success_turn == 0 else "model")
    await _cancel(task)
    restored = _Run(success_turn=success_turn)
    await restored.restore(tmp_path)
    result = await restored.run(1)
    assert result.model_dump() == (await _Run(success_turn=success_turn).run()).model_dump()
    assert restored.verified == ([] if success_turn == 0 else [1, 2])
    assert restored.generated == ([] if success_turn == 0 else [1, 2])


@pytest.mark.asyncio
async def test_completed_model_reply_is_recorded_before_checkpoint_can_prepare(tmp_path):
    source = _Run(stop=("model_read", 0))
    task = asyncio.create_task(source.run())
    await asyncio.wait_for(source.entered.wait(), 5)
    prepared = await source.controller.prepare(CheckpointRequest(**_control(deadline_ts=time.time() + 0.03)))
    assert prepared["phase"] == "preparing"
    assert source.generated == [0]
    source.release.set()
    [record] = await source.commit(tmp_path)
    assert record.episode["next"] == "verify"
    assert record.episode["pending_response"]["id"] == "response-0"
    await _cancel(task)
    restored = _Run()
    await restored.restore(tmp_path)
    result = await restored.run(1)
    assert result.model_dump() == (await _Run().run()).model_dump()
    assert restored.generated == [1, 2]
    assert restored.verified == [0, 1, 2]


@pytest.mark.asyncio
async def test_verifier_reply_does_not_change_exported_boundary_until_resume():
    source = _Run(stop=("verify", 0))
    task = asyncio.create_task(source.run())
    await asyncio.wait_for(source.entered.wait(), 5)
    await source.controller.prepare(CheckpointRequest(**_control()))
    [before] = await source.participant.export(None)
    source.release.set()
    # The result advances to the next boundary and parks before correction generation.
    async with asyncio.timeout(5):
        while not source.participant.legacy_episodes.exported()["run:proof"]["all_attempts"]:
            await asyncio.sleep(0)
    [after] = await source.participant.export(None)
    assert before.episode["next"] == "verify"
    assert before.episode["all_attempts"] == []
    assert after.episode["next"] == "model"
    assert len(after.episode["all_attempts"]) == 1
    assert source.generated == [0]
    await source.controller.resume(CheckpointRequest(**_control()))
    assert (await task).total_turns == 3


@pytest.mark.asyncio
async def test_checkpoint_parked_model_response_retries_after_resume():
    run = _Run()
    run.refuse_once = {"error": {"code": "checkpoint_parked"}}
    task = asyncio.create_task(run.run())
    await asyncio.wait_for(run.entered.wait(), 5)
    prepared = await run.controller.prepare(CheckpointRequest(**_control()))
    assert prepared["phase"] == "prepared"
    assert not task.done()
    assert run.generated == []
    await run.controller.resume(CheckpointRequest(**_control()))
    result = await asyncio.wait_for(task, 5)
    assert result.total_turns == 3
    assert run.generated == [0, 1, 2]


@pytest.mark.asyncio
async def test_retire_cancels_and_fences_the_running_attempt():
    run = _Run(stop=("model", 0))
    task = asyncio.create_task(run.run())
    await asyncio.wait_for(run.entered.wait(), 5)
    await run.controller.retire(RetireRequest(**_control(episode_ids=[EpisodeId(rollout_id="proof")])))
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(StaleAttemptError):
        await run.run()
    assert await run.participant.export(None) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("include_attempts", [False, True])
@pytest.mark.parametrize(
    "success_turn,correction,limit,expected", [(0, True, 2, 1), (2, True, 0, 1), (2, False, 2, 1), (2, True, 2, 3)]
)
async def test_normal_run_stop_rules_and_cookies(enabled, include_attempts, success_turn, correction, limit, expected):
    run = _Run(
        enabled=enabled,
        include_attempts=include_attempts,
        success_turn=success_turn,
        correction=correction,
        max_corrections=limit,
    )
    result = await run.run()
    assert result.total_turns == expected
    assert (len(result.all_attempts) if include_attempts else result.all_attempts) == (
        expected if include_attempts else None
    )
    verify_calls = [call for call in run.calls if call[1] == "/verify"]
    assert all(call[3]["seed"] == "saved" and call[3]["caller"] == "original" for call in verify_calls)
    assert [call[3]["model"] for call in verify_calls] == [f"turn{turn}" for turn in range(expected)]
    if expected > 1:
        correction_input = [call[2] for call in run.calls if call[0] == "policy"][1]
        assert len(correction_input["input"]) == 1
        assert correction_input["input"][0]["role"] == "user"
        assert correction_input["input"][0]["content"] == "correction-1"
        assert correction_input["temperature"] == 0.0
        assert correction_input["top_p"] == 0.8


@pytest.mark.asyncio
async def test_restore_rejects_unexpected_session_payload_and_run_requires_identity():
    run = _Run()
    with pytest.raises(ValueError, match="/run boundary"):
        await run.agent.restore_agent_sessions(
            [RestoredAgentSession("run:r", EpisodeId(rollout_id="r"), {"unexpected": True})]
        )
    assert not run.participant.has_session("run:r")
    with pytest.raises(RolloutIdRequiredError):
        await run.agent.run(_request(), ProofRefinementRunRequest(responses_create_params={"input": "proof"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_key", [None, "direct"])
async def test_direct_responses_does_not_create_a_legacy_session(capture_key):
    run = _Run()
    response = Response()
    result = await run.agent.responses(
        _request(capture_key=capture_key), response, NeMoGymResponseCreateParamsNonStreaming(input="proof")
    )
    assert result.id == "response-0"
    assert "model=turn0" in response.headers["set-cookie"]
    assert await run.participant.export(None) == []
