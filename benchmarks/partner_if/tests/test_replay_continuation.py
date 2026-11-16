# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One public activation delegates prefix decisions to its bound Resources session."""

import copy
import json
from http.cookies import SimpleCookie
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from benchmarks.partner_if.tests.test_agent_sessions import (
    actor_params,
    make_agent,
    provider,
    request_for,
    seed,
)
from benchmarks.partner_if.tests.test_replay import SOURCE, call, probe, response
from nemo_gym.agents import AgentCloseSessionRequest, check_agent_seed_rejected
from nemo_gym.trajectory import VerifierInvocation
from responses_api_agents.ngc_recorded_context import native_agent as adapter


def review_transport(monkeypatch, decisions, reviews, *, corrupt=None, failure=None):
    policy_transport = adapter.http_request

    async def invoke(method, endpoint, **kwargs):
        if not endpoint.endswith("/replay/prefix-review"):
            return await policy_transport(method, endpoint, **kwargs)
        reviews.append((endpoint, copy.deepcopy(kwargs)))
        if failure is not None:
            raise failure
        assert decisions, "Unexpected duplicate or post-decision review"
        next_limit = decisions.pop(0)
        request = adapter.PrefixReviewRequest.model_validate(kwargs["json"])
        digest = adapter.replay_request_sha256(request)
        receipt = f"{request.prefix_index:032x}"
        canonical = adapter.canonical_replay_response(request.response, receipt, digest, masked=next_limit is not None)
        if corrupt == "usage":
            canonical.usage.total_tokens += 1
        data = adapter.PrefixReviewResponse(
            response=canonical,
            reward=0 if next_limit is not None else 1,
            mask_sample=next_limit is not None,
            evaluation_completed=next_limit is None,
            next_parent_turn_limit=next_limit,
            request_sha256=digest,
            receipt_id=receipt,
            ng_verifier_invocations=[
                VerifierInvocation(invocation_id=f"review-{request.prefix_index}", status="completed")
            ],
        ).model_dump(mode="json")
        if corrupt == "receipt":
            data["response"]["partner_if_replay_receipt"]["id"] = "f" * 32
        if corrupt == "request":
            data["request_sha256"] = "e" * 64
            data["response"]["partner_if_replay_receipt"]["request_sha256"] = "e" * 64

        class Result:
            status = 200
            cookies = SimpleCookie("session=updated-resource-cookie")

            async def json(self, **_):
                return data

            def release(self):
                pass

        return Result()

    monkeypatch.setattr(adapter, "http_request", invoke)


@pytest.mark.asyncio
async def test_one_activation_stops_at_the_first_terminal_resources_review(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    policies, reviews = [], []
    provider(
        monkeypatch,
        [
            response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}, "read")),
            response(probe("probe")),
        ],
        policies,
    )
    review_transport(monkeypatch, [2, None], reviews)
    binding = seed()
    binding.tool_accesses[0].headers = {"X-Review-Key": "trusted-review-key", "X-Session-Id": "ignored-id"}
    await agent.seed_agent_session(SimpleNamespace(session={}), binding)
    original = actor_params()
    reply = await agent.responses(request_for(), original)
    repeated = await agent.responses(request_for(), original)
    assert repeated == reply and len(policies) == len(reviews) == 2
    assert reply.usage.total_tokens == 240
    assert reply.partner_if_replay_receipt["id"] == f"{2:032x}"
    assert len(reply.partner_if_replay_history) == 2
    assert len({item["replay_artifacts_dir"] for item in reply.partner_if_replay_history}) == 2
    for index, (endpoint, request) in enumerate(reviews, 1):
        assert endpoint == "http://resources/ng-rollout/rollout-a1/replay/prefix-review"
        assert request["headers"] == {"X-Review-Key": "trusted-review-key", "x-session-id": "step1"}
        assert request["_max_num_tries"] == 1 and request["_max_connection_retries"] == 0
        assert request["_resend_after_disconnect"] is False and request["_retry_other_errors"] is False
        body = adapter.PrefixReviewRequest.model_validate(request["json"])
        assert body.prefix_index == index and body.agent_session_id == "step1"
        assert body.episode_id == seed().episode_id and body.task_id == seed().task_id
        assert body.responses_create_params == original
        assert body.response.usage.total_tokens == index * 120
        directory = adapter.Path(body.response.replay_artifacts_dir)
        metadata = json.loads((directory / "native-step.json").read_text())
        assert metadata["agent_session_id"] == "step1" and metadata["prefix_index"] == index
        assert metadata["episode_id"] == body.episode_id.model_dump(mode="json")
        assert metadata["task_id"] == body.task_id.model_dump(mode="json")
        assert metadata["responses_create_params"] == original.model_dump(exclude_none=True, exclude_unset=True)
        assert adapter.verification_digest(directory) == body.response.replay_verification_sha256
        assert "parent_turn_limit" not in json.dumps(policies[index - 1][2]["json"])
        assert "synthetic-ordering-example" not in json.dumps(policies[index - 1][2]["json"])
    assert reviews[0][1]["cookies"] == {"session": "resource-cookie"}
    assert reviews[1][1]["cookies"] == {"session": "updated-resource-cookie"}
    closed = await agent.close_agent_session(
        request_for(),
        AgentCloseSessionRequest(
            agent_session_id="step1",
            episode_id=seed().episode_id,
        ),
    )
    assert closed.cleanup_confirmed and closed.resources_cookies == {"session": "updated-resource-cookie"}
    assert closed.trajectory is not None
    roots = [item for item in closed.trajectory.invocations if item.parent_invocation_id is None]
    assert [item.invocation_id for item in roots] == ["step1"]
    assert len(closed.trajectory.turns) == len(closed.trajectory.tool_calls) == 2
    assert [item.invocation_id for item in closed.trajectory.verifier_invocations] == ["review-1", "review-2"]
    assert {item.invocation_id for item in closed.trajectory.tool_calls} == {"step1.prefix-1", "step1.prefix-2"}
    assert {item.tool_call_id for item in closed.trajectory.tool_calls} == {"read", "probe"}
    assert {item.harness_session_id for item in closed.trajectory.invocations} == {"step1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", ["receipt", "request", "usage"])
async def test_invalid_resources_review_never_triggers_another_policy_or_judge_attempt(tmp_path, monkeypatch, corrupt):
    agent = make_agent(tmp_path)
    policies, reviews = [], []
    provider(monkeypatch, [response(probe())], policies)
    review_transport(monkeypatch, [2], reviews, corrupt=corrupt)
    await agent.seed_agent_session(SimpleNamespace(session={}), seed())
    reply = await agent.responses(request_for(), actor_params())
    assert reply.status == "incomplete" and "partner_if_replay_receipt" not in (reply.model_extra or {})
    assert await agent.responses(request_for(), actor_params()) == reply
    assert len(policies) == len(reviews) == 1
    state = agent._require_agent_session("step1")
    assert state.reply == reply and state.outer_invocation.status == "failed"
    closed = await agent.close_agent_session(
        request_for(),
        AgentCloseSessionRequest(
            agent_session_id="step1",
            episode_id=seed().episode_id,
        ),
    )
    assert closed.cleanup_confirmed and len(closed.trajectory.turns) == 1


@pytest.mark.asyncio
async def test_ambiguous_review_transport_failure_is_not_retried(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    policies, reviews = [], []
    provider(monkeypatch, [response(probe())], policies)
    review_transport(monkeypatch, [], reviews, failure=TimeoutError("review reply lost"))
    await agent.seed_agent_session(SimpleNamespace(session={}), seed())
    reply = await agent.responses(request_for(), actor_params())
    assert reply.status == "incomplete" and reply.replay_review_error == "TimeoutError"
    assert "partner_if_replay_receipt" not in (reply.model_extra or {})
    assert await agent.responses(request_for(), actor_params()) == reply
    assert len(policies) == len(reviews) == 1
    assert agent._require_agent_session("step1").reply == reply


@pytest.mark.asyncio
async def test_resources_cannot_extend_the_configured_policy_prefix_budget(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    agent.config.replay.max_new_turns = 1
    policies, reviews = [], []
    provider(monkeypatch, [response(probe())], policies)
    review_transport(monkeypatch, [2], reviews)
    await agent.seed_agent_session(SimpleNamespace(session={}), seed())
    reply = await agent.responses(request_for(), actor_params())
    assert reply.status == "incomplete" and reply.replay_review_error == "ValueError"
    assert "partner_if_replay_receipt" not in (reply.model_extra or {})
    assert len(policies) == len(reviews) == 1
    assert agent._require_agent_session("step1").reply == reply


@pytest.mark.parametrize("override", [{"tool_accesses": []}, {"agent_context": None}])
def test_missing_resources_binding_is_rejected_before_inference(tmp_path, override):
    agent = make_agent(tmp_path)
    assert check_agent_seed_rejected(agent.setup_webserver(), seed(**override))
    assert not agent._session_records


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control",
    [
        {"parent_turn_limit": 2},
        {"saved_prefix": str(SOURCE)},
        {"metadata": {"ngc_replay_parent_turn_limit": "2"}},
    ],
)
async def test_public_activation_cannot_override_internal_continuation_controls(tmp_path, control):
    agent = make_agent(tmp_path)
    await agent.seed_agent_session(SimpleNamespace(session={}), seed())
    body = adapter.ReplayActivation.model_validate(actor_params().model_dump(exclude_unset=True) | control)
    with pytest.raises(HTTPException) as rejected:
        await agent.responses(request_for(), body)
    assert rejected.value.status_code == 422
    assert agent._require_agent_session("step1").request_bytes is None
