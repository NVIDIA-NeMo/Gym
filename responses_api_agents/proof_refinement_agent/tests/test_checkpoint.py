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
"""Proof continuations survive disk checkpoints without repeating completed turns."""

import asyncio
import json
import time
from copy import deepcopy
from http.cookies import SimpleCookie
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from nemo_gym._checkpoint import (
    AGENT_EXECUTION_GENERATION_HEADER,
    RESOURCE_REQUEST_ID_HEADER,
    RESOURCE_STATE_REVISION_HEADER,
    AgentBoundaryKind,
    commit_agent_state,
    restore_agent_state,
)
from nemo_gym.rollout_correlation import (
    MODEL_CALL_CAPTURE_OUTCOME_HEADER,
    MODEL_CALL_ID_HEADER,
    current_rollout_id,
    take_checkpoint_parent,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.lineage import InMemoryLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext, reset_token_sink, resolve_parent, set_token_sink
from responses_api_agents.proof_refinement_agent.app import (
    ProofRefinementAgent,
    ProofRefinementAgentConfig,
    ProofRefinementRunRequest,
)


ROLLOUT = "proof-rollout"


def _response(body, *, headers=None, cookies=None):
    response = AsyncMock()
    response.ok = True
    response.headers = headers or {}
    response.cookies = SimpleCookie(cookies or {})
    response.read.return_value = json.dumps(body).encode()
    return response


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


def _request():
    return Request({"type": "http", "headers": [(b"cookie", b"caller=original")], "path": "/run"})


class _Run:
    """Use the registered /run wrapper, replacing only downstream HTTP responses."""

    def __init__(self, *, stop=None, success_turn=2, correction=True, max_corrections=2, include_attempts=True):
        self.stop = stop
        self.success_turn = success_turn
        self.correction = correction
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.parents = []
        self.generated = []
        self.verified = []
        self.capture_outcome = "captured"
        self.capture_outcome_turn = 0
        self.prepare_task = None
        client = MagicMock(spec=ServerClient)
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
                checkpoint_replayable_verify=True,
            ),
            server_client=client,
        )
        self.participant = self.agent.checkpoint_participant()
        app = self.agent.setup_webserver()
        self.endpoint = next(route.endpoint for route in app.routes if route.path == "/run")
        if stop == ("finished", 2):
            original_commit = self.participant.commit_boundary

            async def commit(execution, boundary):
                if boundary.agent_state["finished"]:
                    self.prepare_task = asyncio.create_task(self.participant.prepare(time.time() + 2))
                    await asyncio.sleep(0)
                    self.entered.set()
                await original_commit(execution, boundary)

            self.participant.commit_boundary = commit

    async def pause(self, stage, turn):
        if self.stop == (stage, turn):
            self.entered.set()
            await self.release.wait()

    async def post(self, *, server_name, url_path, json, cookies, headers=None):
        call = deepcopy(
            {"server": server_name, "path": url_path, "body": json, "cookies": cookies, "headers": headers}
        )
        self.calls.append(call)
        if url_path == "/seed_session":
            return _response({}, cookies={"session": "seeded"}, headers={RESOURCE_STATE_REVISION_HEADER: "1"})
        if server_name == "proof-agent":
            text = json["input"][0]["content"]
            turn = int(text.rsplit("-", 1)[1])
            await self.pause("model", turn)
            self.parents.append(take_checkpoint_parent())
            self.generated.append(turn)
            outcome = self.capture_outcome if turn == self.capture_outcome_turn else "captured"
            model = _model_response(turn)
            model_headers = {MODEL_CALL_CAPTURE_OUTCOME_HEADER: outcome}
            if outcome == "captured":
                model_headers[MODEL_CALL_ID_HEADER] = f"call-{turn}"
            elif outcome == "no_generation":
                model.update(output=[], status="incomplete", incomplete_details={"reason": "content_filter"})
            call["capture_key"] = current_rollout_id()
            return _response(model, cookies={"model": f"turn-{turn}"}, headers=model_headers)
        assert server_name == "lean" and url_path == "/verify"
        turn = json["turn_index"]
        await self.pause("verify", turn)
        self.verified.append(turn)
        success = turn == self.success_turn
        return _response(
            {
                "responses_create_params": json["responses_create_params"],
                "response": json["response"],
                "reward": float(success),
                "proof_status": "completed" if success else "failed",
                "needs_correction": not success,
                "error_feedback": None if success else f"error-{turn}",
                "correction_prompt": f"prompt-{turn + 1}" if self.correction and not success else None,
            },
            cookies={"verified": f"turn-{turn}"},
            headers={RESOURCE_STATE_REVISION_HEADER: str(turn + 2)},
        )

    async def run(self, attempt=0):
        body = ProofRefinementRunRequest.model_validate(
            {
                "_ng_rollout_id": ROLLOUT,
                "_ng_attempt_index": attempt,
                "responses_create_params": {
                    "input": [{"role": "user", "content": "prompt-0"}],
                    "model": "proof-model",
                    "temperature": 0.25,
                    "top_p": 0.8,
                },
                "verifier_metadata": {"theorem": "example"},
            }
        )
        return await self.endpoint(request=_request(), body=body)

    async def save(self, directory, attempt=0):
        task = asyncio.create_task(self.run(attempt))
        await asyncio.wait_for(self.entered.wait(), 2)
        report = (
            await self.prepare_task
            if self.prepare_task is not None
            else await self.participant.prepare(time.time() + 2, allow_model_wait_boundary=True)
        )
        assert report["ready_to_commit"] is True
        assert report["parked_with_boundary"] == 1
        boundary = self.participant.resolve(ROLLOUT, attempt).boundary.model_copy(deep=True)
        assert commit_agent_state(self.participant, directory, checkpoint_id="proof-checkpoint")["records"] == 1
        await self.participant.retire(ROLLOUT, attempt)
        await asyncio.gather(task, return_exceptions=True)
        return boundary

    async def restore(self, directory):
        assert restore_agent_state(self.participant, directory)["records"] == 1
        await self.participant.resume()


class _CapturedRun(_Run):
    """Resolve real capture ancestry for the simulated model's proof outputs."""

    def __init__(self, *, ledger=None, **kwargs):
        super().__init__(**kwargs)
        self.ledger = ledger if ledger is not None else InMemoryLineageStore()
        self.admissions = []

    async def post(self, **kwargs):
        response = await super().post(**kwargs)
        if kwargs["server_name"] != "proof-agent":
            return response
        turn = self.generated[-1]
        source_capture_key, parent_model_call_id = self.parents[-1]
        capture_key = current_rollout_id()
        context = CaptureContext(
            rollout_id=capture_key,
            model_call_id=f"call-{turn}",
            token_sink=None,
            lineage_store=self.ledger,
            external_staging=True,
            source_capture_key=source_capture_key,
            explicit_parent_call_id=parent_model_call_id,
        )
        token = set_token_sink(context)
        try:
            await resolve_parent(kwargs["json"]["input"])
        finally:
            reset_token_sink(token)
        if context.capture_admission is None:
            response.headers[MODEL_CALL_CAPTURE_OUTCOME_HEADER] = context.capture_outcome
            response.headers.pop(MODEL_CALL_ID_HEADER, None)
        else:
            self.admissions.append(context.capture_admission)
            self.ledger.index.for_rollout(capture_key).record(
                f"call-{turn}",
                kwargs["json"]["input"] + _model_response(turn)["output"],
                cum_tokens=[turn + 1],
                digest=f"proof-{turn}",
            )
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [("model", 0), ("verify", 0), ("model", 1), ("finished", 2)])
async def test_restored_proof_capture_keeps_independent_correction_roots(tmp_path, stop):
    baseline = _CapturedRun()
    expected = await baseline.run()
    original = _CapturedRun(stop=stop)
    await original.save(tmp_path)
    replacement = _CapturedRun(ledger=original.ledger)
    await replacement.restore(tmp_path)
    result = await replacement.run(attempt=1)
    assert result.model_dump() == expected.model_dump()
    assert len(baseline.admissions) == 3
    assert len(original.admissions) + len(replacement.admissions) == 3
    for admission in baseline.admissions + original.admissions + replacement.admissions:
        assert admission.mode == "text"
        assert admission.parent_call_id is None
    assert (await replacement.ledger.manifest(f"{ROLLOUT}-a1"))["failures"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop", "index", "turn", "kind", "generated", "verified"),
    [
        (("model", 0), 0, 0, AgentBoundaryKind.TURN_COMPLETE, [0, 1, 2], [0, 1, 2]),
        (("verify", 0), 1, 0, AgentBoundaryKind.PENDING_MODEL, [1, 2], [0, 1, 2]),
        (("model", 1), 2, 1, AgentBoundaryKind.TURN_COMPLETE, [1, 2], [1, 2]),
        (("finished", 2), 6, 2, AgentBoundaryKind.TURN_COMPLETE, [], []),
    ],
)
async def test_disk_restore_matches_uninterrupted_proof(tmp_path, stop, index, turn, kind, generated, verified):
    baseline = await _Run().run()
    original = _Run(stop=stop)
    boundary = await original.save(tmp_path)
    assert (boundary.boundary_index, boundary.turn_index, boundary.boundary_kind) == (index, turn, kind)
    assert boundary.agent_state["cookies"]["session"] == "seeded"
    assert boundary.resource_state_revisions == {"lean": (4 if index == 6 else turn + 1)}
    assert len(boundary.agent_state["all_attempts"]) == (3 if index == 6 else turn)
    if kind == AgentBoundaryKind.PENDING_MODEL:
        assert boundary.pending_model.response == _model_response(0)
        assert boundary.pending_model.model_server_cookies["model"] == "turn-0"
        pending_verify = next(call for call in original.calls if call["path"] == "/verify")
        assert boundary.pending_model.resource_request_id == pending_verify["headers"][RESOURCE_REQUEST_ID_HEADER]

    replacement = _Run()
    await replacement.restore(tmp_path)
    result = await replacement.run(attempt=1)
    assert result.model_dump() == baseline.model_dump()
    assert replacement.generated == generated
    assert replacement.verified == verified
    assert all(call["path"] != "/seed_session" for call in replacement.calls)
    assert [attempt["generation"] for attempt in result.all_attempts] == ["proof-0", "proof-1", "proof-2"]
    assert [attempt["turn_index"] for attempt in result.all_attempts] == [0, 1, 2]
    assert replacement.participant.resolve(ROLLOUT, 1).boundary.boundary_index == 6
    if generated:
        assert replacement.parents == [(None, None)] * len(generated)
    if kind == AgentBoundaryKind.PENDING_MODEL:
        resumed_verify = next(call for call in replacement.calls if call["path"] == "/verify")
        assert resumed_verify["headers"] == pending_verify["headers"]
        assert resumed_verify["body"] == pending_verify["body"] | {"_ng_attempt_index": 1}
        assert resumed_verify["cookies"] == pending_verify["cookies"]
    if index == 6:
        receipt = replacement.participant.completion_receipt(ROLLOUT, 1)
        assert receipt.manifest_capture_key == ROLLOUT
        assert receipt.terminal_model_call_id == "call-2"


@pytest.mark.asyncio
async def test_recheckpoint_before_restored_verification_keeps_original_parent(tmp_path):
    first = _Run(stop=("verify", 0))
    original = await first.save(tmp_path / "first")
    second = _Run(stop=("verify", 0))
    await second.restore(tmp_path / "first")
    repeated = await second.save(tmp_path / "second", attempt=1)
    assert repeated.boundary_index == original.boundary_index
    assert repeated.attempt_index == 1
    assert repeated.pending_model == original.pending_model
    assert repeated.last_committed_model_capture_key == ROLLOUT
    assert repeated.agent_state["all_attempts"] == []
    assert second.generated == []

    third = _Run()
    await third.restore(tmp_path / "second")
    result = await third.run(attempt=2)
    assert result.reward == 1
    assert result.total_turns == 3
    assert third.generated == [1, 2]
    assert third.parents == [(None, None), (None, None)]
    assert third.participant.completion_receipt(ROLLOUT, 2).manifest_capture_key == f"{ROLLOUT}-a2"


@pytest.mark.asyncio
async def test_recheckpoint_before_correction_generation_keeps_saved_capture_coordinate(tmp_path):
    first = _CapturedRun(stop=("model", 1))
    original = await first.save(tmp_path / "first")
    second = _CapturedRun(stop=("model", 1), ledger=first.ledger)
    await second.restore(tmp_path / "first")
    repeated = await second.save(tmp_path / "second", attempt=1)
    assert repeated.last_committed_model_capture_key == original.last_committed_model_capture_key == ROLLOUT
    assert repeated.last_committed_model_call_id == original.last_committed_model_call_id == "call-0"
    assert repeated.agent_state == original.agent_state
    assert second.generated == []
    third = _CapturedRun(ledger=second.ledger)
    await third.restore(tmp_path / "second")
    result = await third.run(attempt=2)
    assert result.reward == 1
    assert third.generated == [1, 2]
    assert all(admission.mode == "text" for admission in third.admissions)


@pytest.mark.asyncio
async def test_verification_result_is_consumed_only_after_checkpoint_resume():
    harness = _Run(stop=("verify", 0))
    task = asyncio.create_task(harness.run())
    try:
        await asyncio.wait_for(harness.entered.wait(), 2)
        report = await harness.participant.prepare(time.time() + 2)
        assert report["ready_to_commit"] is True
        execution = harness.participant.resolve(ROLLOUT, 0)
        saved = execution.boundary.model_copy(deep=True)
        harness.release.set()
        async with asyncio.timeout(2):
            while not harness.verified:
                await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        assert execution.boundary == saved
        assert saved.agent_state["all_attempts"] == []
        assert "verified" not in saved.agent_state["cookies"]
        await harness.participant.resume()
        result = await asyncio.wait_for(task, 2)
        assert harness.verified == [0, 1, 2]
        assert [item["turn_index"] for item in result.all_attempts] == [0, 1, 2]
    finally:
        await harness.participant.retire(ROLLOUT, 0)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_attempts", [True, False])
async def test_noncheckpoint_run_preserves_results_without_capture_headers(include_attempts):
    expected = await _Run(include_attempts=include_attempts).run()
    ordinary = _Run(include_attempts=include_attempts)
    ordinary.agent._checkpoint_participant = None
    ordinary.agent.config.checkpoint_replayable_verify = False

    async def post(**kwargs):
        response = await ordinary.post(**kwargs)
        response.headers = {}
        return response

    ordinary.agent.server_client.post = AsyncMock(side_effect=post)
    result = await ordinary.run()
    assert result.model_dump() == expected.model_dump()
    assert ordinary.generated == ordinary.verified == [0, 1, 2]
    assert all(call["headers"] is None for call in ordinary.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("success_turn", "correction", "limit", "expected"), [(0, True, 3, 1), (9, False, 3, 1), (9, True, 2, 3)]
)
async def test_stop_rules_and_cookie_propagation(success_turn, correction, limit, expected):
    harness = _Run(success_turn=success_turn, correction=correction, max_corrections=limit)
    result = await harness.run()
    assert result.total_turns == expected
    assert harness.generated == harness.verified == list(range(expected))
    assert len(result.all_attempts) == expected
    seed = harness.calls[0]
    assert seed["cookies"] == {"caller": "original"}
    assert seed["headers"][RESOURCE_REQUEST_ID_HEADER]
    for call in harness.calls[1:]:
        assert call["cookies"]["session"] == "seeded"
        if call["server"] == "proof-agent":
            assert call["headers"][AGENT_EXECUTION_GENERATION_HEADER] == "1"
            assert call["body"]["temperature"] == 0.25
            assert call["body"]["top_p"] == 0.8
        else:
            turn = call["body"]["turn_index"]
            assert call["cookies"]["model"] == f"turn-{turn}"
            assert call["body"]["verifier_metadata"] == {"theorem": "example"}


@pytest.mark.asyncio
@pytest.mark.parametrize("include_attempts", [True, False])
async def test_no_generation_masks_sample_without_verifying_failed_turn(include_attempts):
    harness = _Run(include_attempts=include_attempts)
    harness.capture_outcome = "no_generation"
    harness.capture_outcome_turn = 1
    result = await harness.run()
    assert result.reward == 0
    assert result.mask_sample is True
    assert result.failure_kind == "agent_no_generation"
    assert result.total_turns == 2
    assert harness.verified == [0]
    assert ([item["turn_index"] for item in result.all_attempts] if include_attempts else result.all_attempts) == (
        [0] if include_attempts else None
    )
    boundary = harness.participant.resolve(ROLLOUT, 0).boundary
    assert boundary.boundary_index == 2
    assert boundary.last_committed_model_call_id == "call-0"


@pytest.mark.asyncio
async def test_capture_failure_does_not_commit_generated_proof_or_verify():
    harness = _Run()
    harness.capture_outcome = "capture_failed"
    commits = AsyncMock(wraps=harness.participant.commit_boundary)
    harness.participant.commit_boundary = commits
    with pytest.raises(RuntimeError, match="without durable token capture"):
        await harness.run()
    assert harness.verified == []
    assert harness.participant.resolve(ROLLOUT, 0) is None
    assert commits.await_count == 1
    boundary = commits.await_args.args[1]
    assert boundary.boundary_index == 0
    assert boundary.pending_model is None
    assert boundary.last_committed_model_call_id is None


@pytest.mark.asyncio
async def test_responses_forwards_capture_headers_and_cookie_values():
    harness = _Run()
    headers = {MODEL_CALL_ID_HEADER: "ledger-call", MODEL_CALL_CAPTURE_OUTCOME_HEADER: "captured"}
    harness.agent.server_client.post = AsyncMock(
        return_value=_response(_model_response(0), headers=headers, cookies={"model": "saved"})
    )
    response = Response()
    params = ProofRefinementRunRequest(responses_create_params={"input": "prove it"}).responses_create_params
    result = await harness.agent.responses(_request(), response, params)
    assert result.id == "response-0"
    assert all(response.headers[key] == value for key, value in headers.items())
    cookies = SimpleCookie(response.headers["set-cookie"])
    assert cookies["model"].value == "saved"
    call = harness.agent.server_client.post.await_args.kwargs
    assert call["server_name"] == "policy"
    assert call["cookies"] == {"caller": "original"}
    assert call["json"].input[0].role == "user"
    assert call["json"].input[0].content == "prove it"
    assert params.input == "prove it"
