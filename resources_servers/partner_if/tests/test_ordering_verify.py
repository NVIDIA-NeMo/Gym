# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real immutable prefixes with controlled grades; no live models or semantic calibration."""

import asyncio
import copy
import json
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from benchmarks.partner_if.tests.fixture_data import BUNDLE, PARAMS, PROBE, SOURCE, TASK
from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId, rollout_context
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from nemo_gym.verification import JudgeError
from resources_servers.partner_if import app
from resources_servers.partner_if.engine import aggregate
from resources_servers.partner_if.grading_contract import RawJudgment
from resources_servers.partner_if.grading_service import GradeResponse
from resources_servers.partner_if.replay_protocol import (
    PrefixReviewRequest,
    canonical_replay_response,
    replay_request_sha256,
)
from resources_servers.partner_if.replay_receipts import ReplayReceiptStore
from responses_api_agents.ngc_recorded_context import native_agent
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, actor_identity
from responses_api_agents.ngc_recorded_context.saved_prefix import verification_digest


def call(name, arguments, identity="tool"):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def probe(identity="probe"):
    return call("Bash", {"command": f"python3 -I {PROBE} --uri '{BUNDLE['sibling']['input_uri']}'"}, identity)


def provider_response(calls=(), text=None):
    return {
        "id": "policy-1",
        "model": "test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": list(calls),
                },
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


async def native_capture(tmp_path, monkeypatch, *, calls=(), text=None, error=None):
    config = native_agent.NativeReplayAgentConfig.model_validate(
        {
            "name": "replay",
            "host": "127.0.0.1",
            "port": 9999,
            "entrypoint": "native_agent.py",
            "model": "test",
            "model_server": {"type": "responses_api_models", "name": "policy"},
            "resources_server": {"type": "resources_servers", "name": "resources"},
            "replay": ReplayConfig(
                package_dir=SOURCE / "assets", artifacts_dir=tmp_path, coverage_profile="expanded-v4"
            ),
        }
    )
    agent = native_agent.NativeReplayAgent(config=config, server_client=MagicMock(spec=ServerClient))
    seen = []

    async def request(*args, **kwargs):
        seen.append(kwargs["json"])
        if error is not None:
            raise error

        class Response:
            status = 200
            cookies = SimpleCookie()

            async def json(self, **_):
                return provider_response(calls, text)

            def release(self):
                pass

        return Response()

    monkeypatch.setattr(native_agent, "http_request", request)
    monkeypatch.setattr(native_agent.NativeReplayAgent, "resolve_model_base_url", lambda *args: "http://policy/v1")
    seed = AgentSeedSessionRequest(
        agent_session_id="step1",
        episode_id=EpisodeId(rollout_id="test", attempt=1),
        task_id=TaskId(taskset="fixture", task_id="pif_029"),
    )
    state = native_agent.ReplaySession(request=seed, outer_agent_session_id=seed.agent_session_id)
    req = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "session": {"agent_session_id": "step1"},
            "path_params": {"rollout_id": "test-a1"},
        }
    )
    reply = await agent._execute_step(
        req, state, native_agent.ReplayActivation.model_validate(PARAMS | {"parent_turn_limit": 1})
    )
    assert len(seen) == 1
    body = app.VerifyRequest.model_validate(TASK | {"response": reply.model_dump(mode="json")})
    return body, Path(reply.replay_artifacts_dir)


def server(tmp_path, monkeypatch, *, passed=True, opted_in=True, max_calls=8, max_prefixes=6, observed=False):
    calls = []

    async def semantic_fixture(contract, evidence, **kwargs):
        calls.append(evidence)
        await asyncio.sleep(0.01)
        judgments = [
            RawJudgment(
                id=item.id, passed=passed, reason="Controlled ordering verdict.", evidence="Synthetic public fixture."
            )
            for item in contract.checks
        ]
        return GradeResponse(grading_result=aggregate(contract, judgments))

    monkeypatch.setattr(app, "grade", semantic_fixture)
    result = app.PartnerIFResourcesServer(
        config=app.PartnerIFResourcesServerConfig(
            name="resources",
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            accept_verification_input=True,
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            ordering_artifacts_dir=tmp_path if opted_in else None,
            ordering_max_policy_calls=max_calls,
            ordering_max_prefixes=max_prefixes,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    result.server_client.global_config_dict = {"observability_enabled": observed}
    return result, calls


def review_request(body):
    task_data = body.model_dump(mode="json", exclude={"responses_create_params", "response"}, exclude_none=True)
    return PrefixReviewRequest(
        episode_id=EpisodeId(rollout_id="test", attempt=1),
        task_id=TaskId(taskset="fixture", task_id="pif_029"),
        agent_session_id="step1",
        prefix_index=1,
        task_data=task_data,
        responses_create_params=body.responses_create_params,
        response=body.response,
    )


def request(payload=None, session="resources-session"):
    value = Request(
        {"type": "http", "method": "POST", "path": "/verify", "headers": [], "session": {SESSION_ID_KEY: session}}
    )
    value._body = json.dumps(payload or {}).encode()
    return value


async def seeded_review(resource, prefix, incoming):
    session_id = incoming.session[SESSION_ID_KEY]
    seed = ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=prefix.episode_id,
        task_id=prefix.task_id,
        task_data=prefix.task_data,
        responses_create_params=prefix.model_dump(mode="json")["responses_create_params"],
    )
    await resource.seed_session(seed, request=incoming)
    return await resource.prefix_review(prefix, incoming)


async def final_verify(resource, prefix, review, *, session="resources-session", update=None):
    payload = prefix.task_data | {
        "responses_create_params": prefix.responses_create_params.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        ),
        "response": review.response.model_dump(mode="json"),
    }
    if update:
        update(payload)
    return await resource.verify(app.VerifyRequest.model_validate(payload), request(payload, session))


@pytest.mark.asyncio
async def test_pending_review_returns_bounded_control_without_changing_frozen_actor_inputs(tmp_path, monkeypatch):
    body, _ = await native_capture(
        tmp_path, monkeypatch, calls=[call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})]
    )
    resource, grades = server(tmp_path, monkeypatch)
    prefix = review_request(body)
    result = await seeded_review(resource, prefix, request())
    assert len(grades) == 1
    assert actor_identity({"input": grades[0].context}) == actor_identity({"input": PARAMS["input"]})
    assert result.mask_sample and not result.evaluation_completed and result.reward == 0
    assert result.replay_diagnostics["termination"] == "ordering_pending"
    assert result.next_parent_turn_limit == 2
    assert result.response == canonical_replay_response(
        prefix.response, result.receipt_id, replay_request_sha256(prefix), masked=True
    )
    assert result.response.status == "incomplete"
    assert "next_agent_request" not in result.model_dump()
    assert "parallel_tool_calls" not in json.loads(prefix.model_dump_json())["responses_create_params"]
    assert (
        PrefixReviewRequest.model_validate_json(prefix.model_dump_json()).responses_create_params.model_fields_set
        == prefix.responses_create_params.model_fields_set
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passed,calls",
    [
        (True, [probe()]),
        (False, [call("Write", {"file_path": "/workspace/logs/answer.txt", "content": "Publish now."})]),
        (
            False,
            [probe(), call("Write", {"file_path": "/workspace/logs/answer.txt", "content": "Publish now."}, "draft")],
        ),
    ],
)
async def test_decisive_prefix_and_final_verify_share_one_judge_receipt(tmp_path, monkeypatch, passed, calls):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=calls)
    resource, grades = server(tmp_path, monkeypatch, passed=passed)
    prefix = review_request(body)
    result = await seeded_review(resource, prefix, request())
    assert len(grades) == 1 and result.reward == float(passed)
    assert not result.mask_sample and result.evaluation_completed and result.next_parent_turn_limit is None
    assert result.replay_diagnostics["termination"] == ("ordering_passed" if passed else "ordering_failed")
    final = await final_verify(resource, prefix, result)
    assert len(grades) == 1 and final.grading_result.model_dump(mode="json") == result.grading_result
    assert final.response == result.response and final.reward == result.reward and not final.mask_sample


@pytest.mark.asyncio
async def test_identical_concurrent_retries_and_restart_reuse_exact_persisted_observations(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch, observed=True)
    prefix = review_request(body)
    other_worker = app.PartnerIFResourcesServer(config=resource.config, server_client=resource.server_client)
    with rollout_context("test-a1"):
        results = await asyncio.gather(
            seeded_review(resource, prefix, request()),
            seeded_review(other_worker, prefix, request()),
            seeded_review(resource, prefix, request()),
        )
    assert len(grades) == 1 and results[0] == results[1] == results[2]
    assert len(results[0].ng_verifier_invocations) == 1
    assert results[0].ng_verifier_invocations[0].grading_record is None
    restarted = app.PartnerIFResourcesServer(config=resource.config, server_client=resource.server_client)
    retried = await seeded_review(restarted, prefix, request())
    assert retried == results[0] and len(grades) == 1
    final = await final_verify(restarted, prefix, retried)
    assert final.response == retried.response and not final.mask_sample and len(grades) == 1


@pytest.mark.asyncio
async def test_crashed_dispatch_marker_masks_without_hidden_regrade(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch)
    prefix = review_request(body)
    store = ReplayReceiptStore(tmp_path)
    scope = store.scope("resources-session", prefix)
    identity = store.receipt_id(scope)
    store.write(
        identity,
        {
            "state": "in_progress",
            "receipt_id": identity,
            "request_sha256": replay_request_sha256(prefix),
            "scope": scope,
            "request": prefix.model_dump(mode="json"),
        },
    )
    result = await seeded_review(resource, prefix, request())
    assert not grades and result.mask_sample and not result.evaluation_completed
    assert result.next_parent_turn_limit is None and result.response.status == "incomplete"
    assert result.failure_kind == "partner_if:replay_review_uncertain"
    assert await seeded_review(resource, prefix, request()) == result
    final = await final_verify(resource, prefix, result)
    assert final.mask_sample and final.failure_kind == result.failure_kind and not grades


@pytest.mark.asyncio
async def test_judge_cancellation_leaves_uncertain_marker_and_retry_never_regrades(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, _ = server(tmp_path, monkeypatch)
    entered = asyncio.Event()
    grades = []

    async def blocked(*args, **kwargs):
        grades.append(True)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "grade", blocked)
    prefix = review_request(body)
    running = asyncio.create_task(seeded_review(resource, prefix, request()))
    await entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    result = await seeded_review(resource, prefix, request())
    assert len(grades) == 1 and result.mask_sample and result.failure_kind == "partner_if:replay_review_uncertain"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["final", "technical", "prefix_cap", "policy_cap"])
async def test_resource_terminal_masks_are_bound_before_final_verify(tmp_path, monkeypatch, terminal):
    options = (
        {"text": "Still investigating."}
        if terminal == "final"
        else {"error": TimeoutError("scripted provider fault")}
        if terminal == "technical"
        else {"calls": [call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})]}
    )
    body, _ = await native_capture(tmp_path, monkeypatch, **options)
    resource, grades = server(
        tmp_path,
        monkeypatch,
        max_calls=1 if terminal == "policy_cap" else 8,
        max_prefixes=1 if terminal == "prefix_cap" else 6,
    )
    prefix = review_request(body)
    result = await seeded_review(resource, prefix, request())
    assert len(grades) == (0 if terminal == "technical" else 1)
    assert result.mask_sample and result.next_parent_turn_limit is None and result.response.status == "incomplete"
    final = await final_verify(resource, prefix, result)
    assert final.response == result.response and final.mask_sample
    assert final.failure_kind == result.failure_kind


@pytest.mark.asyncio
async def test_decisive_last_allowed_call_wins_over_both_caps(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch, max_calls=1, max_prefixes=1)
    result = await seeded_review(resource, review_request(body), request())
    assert len(grades) == 1 and result.reward == 1 and not result.mask_sample
    assert result.next_parent_turn_limit is None and result.response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "usage",
        "output",
        "status",
        "receipt_id",
        "request_hash",
        "params",
        "rubric",
        "task_extra",
        "cookie",
        "capture_files",
        "extra_field",
        "param_default",
    ],
)
async def test_final_receipt_tampering_is_masked_without_another_judge(tmp_path, monkeypatch, fault):
    body, root = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch)
    prefix = review_request(body)
    prefix.task_data["full_original_metadata"] = {"binding": "required"}
    result = await seeded_review(resource, prefix, request())

    def change(payload):
        if fault == "usage":
            payload["response"]["usage"]["total_tokens"] += 1
        elif fault == "output":
            payload["response"]["output"] = []
        elif fault == "status":
            payload["response"]["status"] = "incomplete"
        elif fault == "receipt_id":
            payload["response"]["partner_if_replay_receipt"]["id"] = "0" * 32
        elif fault == "request_hash":
            payload["response"]["partner_if_replay_receipt"]["request_sha256"] = "0" * 64
        elif fault == "params":
            payload["responses_create_params"]["temperature"] = 0.5
        elif fault == "param_default":
            payload["responses_create_params"]["parallel_tool_calls"] = True
        elif fault == "rubric":
            payload["verifier_metadata"]["partner_if"]["rubrics"][0]["constraint"] += " changed"
        elif fault == "task_extra":
            payload["full_original_metadata"] = {"binding": "changed"}
        elif fault == "capture_files":
            (root / "events.json").write_text("[]")
        elif fault == "extra_field":
            payload["unexpected"] = "unbound"

    final = await final_verify(
        resource, prefix, result, session="other-session" if fault == "cookie" else "resources-session", update=change
    )
    assert len(grades) == 1 and final.mask_sample and not final.evaluation_completed
    assert final.failure_kind == "partner_if:replay_invalid_receipt"


@pytest.mark.asyncio
async def test_prefix_replay_cannot_fall_back_to_ordinary_judge_without_terminal_receipt(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch)
    result = await resource.verify(body, request())
    assert not grades and result.mask_sample and result.failure_kind == "partner_if:replay_invalid_receipt"


@pytest.mark.asyncio
async def test_same_prefix_with_different_request_conflicts_without_regrade(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch)
    prefix = review_request(body)
    await seeded_review(resource, prefix, request())
    altered = prefix.model_copy(deep=True)
    altered.task_data["extra_metadata"] = "changed"
    with pytest.raises(HTTPException, match="durable typed Resources seed"):
        await resource.prefix_review(altered, request())
    assert len(grades) == 1


@pytest.mark.asyncio
async def test_judge_outage_becomes_durable_masked_receipt(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch)
    attempts = []

    async def failed_judge(*args, **kwargs):
        attempts.append(True)
        raise JudgeError("scripted judge unavailable")

    monkeypatch.setattr(app, "grade", failed_judge)
    prefix = review_request(body)
    result = await seeded_review(resource, prefix, request())
    assert len(attempts) == 1 and not grades and result.mask_sample
    assert result.next_parent_turn_limit is None and result.response.status == "incomplete"
    final = await final_verify(resource, prefix, result)
    assert final.mask_sample and len(attempts) == 1
    assert final.model_extra["_ng_failure_class"] == "judge_failed"
    assert final.model_extra["_ng_failure_events"]


@pytest.mark.asyncio
async def test_plain_live_agent_rows_keep_ordinary_shared_grading(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    body.verification_input = app.LiveAgentCapture.model_validate(body.response.verification_input)
    for key in list(body.response.model_extra):
        body.response.model_extra.pop(key)
    resource, grades = server(tmp_path, monkeypatch)
    result = await resource.verify(body)
    assert result.reward == 1 and not result.mask_sample and len(grades) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "capture",
        "input",
        "task",
        "criterion",
        "declaration",
        "digest",
        "events",
        "output",
        "outside",
        "symlink",
        "saved_digest",
        "model",
        "receipts_rehashed",
        "api_symlink",
    ],
)
async def test_capture_identity_and_immutable_receipt_faults_mask_before_judging(tmp_path, monkeypatch, fault):
    body, root = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    if fault == "capture":
        captured = copy.deepcopy(body.response.verification_input)
        captured["input"] = []
        body.response.verification_input = captured
    elif fault == "input":
        body.responses_create_params.input = []
    elif fault == "task":
        body.verifier_metadata.partner_if.task_id = "another-task"
    elif fault == "criterion":
        body.verifier_metadata.partner_if.rubrics[0].id = "another-criterion"
    elif fault == "declaration":
        body.response.replay_ordering_contract = body.response.replay_ordering_contract | {"probe_namespace": "other"}
    elif fault == "digest":
        body.response.replay_verification_sha256 = "0" * 64
    elif fault == "events":
        (root / "events.json").write_text("[]")
    elif fault == "output":
        body.response.output = []
    elif fault == "outside":
        body.response.replay_artifacts_dir = str(tmp_path.parent)
    elif fault == "symlink":
        outside = tmp_path.parent / (tmp_path.name + "-outside.json")
        outside.write_text((root / "events.json").read_text())
        (root / "events.json").unlink()
        (root / "events.json").symlink_to(outside)
    elif fault == "saved_digest":
        body.response.saved_prefix_sha256 = "0" * 64
    elif fault == "model":
        body.response.model = "another-model"
    elif fault == "receipts_rehashed":
        captured = json.loads((root / "ordering-verification-input.json").read_text())
        captured["artifacts"]["replay_receipts"]["text"] = "[]"
        (root / "ordering-verification-input.json").write_text(json.dumps(captured))
        body.response.verification_input = captured
        body.response.replay_verification_sha256 = verification_digest(root)
    elif fault == "api_symlink":
        outside = tmp_path.parent / (tmp_path.name + "-api.json")
        original = root / "call-01.request.json"
        outside.write_text(original.read_text())
        original.unlink()
        original.symlink_to(outside)
    resource, grades = server(tmp_path, monkeypatch)
    result = await seeded_review(resource, review_request(body), request())
    assert not grades and result.mask_sample and result.next_parent_turn_limit is None
    assert result.replay_diagnostics["termination"] == "invalid_capture"
    assert result.replay_artifacts_dir == body.response.replay_artifacts_dir


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["episode", "task", "agent", "prefix", "tools", "temperature"])
async def test_review_runtime_identity_and_all_actor_params_are_checked_before_judging(tmp_path, monkeypatch, fault):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch)
    prefix = review_request(body)
    if fault == "episode":
        prefix.episode_id = EpisodeId(rollout_id="other", attempt=1)
    elif fault == "task":
        prefix.task_id = TaskId(taskset="other-taskset", task_id="different-task")
    elif fault == "agent":
        prefix.agent_session_id = "another-agent"
    elif fault == "prefix":
        # A prior review must authorize every noninitial prefix.
        prefix.prefix_index = 2
        with pytest.raises(HTTPException, match="preceding review"):
            await seeded_review(resource, prefix, request())
        assert not grades
        return
    elif fault == "tools":
        prefix.responses_create_params.tools = []
    elif fault == "temperature":
        prefix.responses_create_params.temperature = 0.5
    result = await seeded_review(resource, prefix, request())
    assert not grades and result.mask_sample and result.next_parent_turn_limit is None
    assert result.failure_kind == "partner_if:replay_invalid_capture"


@pytest.mark.asyncio
async def test_replay_cache_observer_has_no_false_grade_and_ordinary_route_keeps_programmatic_records(
    tmp_path, monkeypatch
):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch, observed=True)
    prefix = review_request(body)
    with rollout_context("test-a1"):
        review = await seeded_review(resource, prefix, request())
    application = resource.setup_webserver()
    endpoint = next(route.endpoint for route in application.routes if getattr(route, "path", None) == "/verify")
    payload = prefix.task_data | {
        "responses_create_params": prefix.model_dump(mode="json")["responses_create_params"],
        "response": review.response.model_dump(mode="json"),
    }
    with rollout_context("test-a1"):
        final = await endpoint(request(payload), app.VerifyRequest.model_validate(payload))
    assert len(grades) == 1 and final.ng_verifier_invocations[0].grading_record is None
    assert final.ng_verifier_invocations[0].invocation_id != review.ng_verifier_invocations[0].invocation_id
    plain = body.model_copy(deep=True)
    plain.verification_input = app.LiveAgentCapture.model_validate(plain.response.verification_input)
    plain.response.model_extra.clear()
    with rollout_context("test-a1"):
        ordinary = await endpoint(request(), plain)
    assert len(grades) == 2 and ordinary.ng_verifier_invocations[0].grading_record is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["episode", "attempt", "task", "task_data", "params", "unseeded_cookie"])
async def test_typed_seed_identity_is_required_even_without_rollout_context(tmp_path, monkeypatch, field):
    body, root = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch)
    prefix = review_request(body)
    original = prefix.model_copy(deep=True)
    await resource.seed_session(
        ResourcesSeedSessionRequest(
            resources_session_id="resources-session",
            episode_id=prefix.episode_id,
            task_id=prefix.task_id,
            task_data=prefix.task_data,
            responses_create_params=prefix.model_dump(mode="json")["responses_create_params"],
        ),
        request=request(),
    )
    if field == "episode":
        prefix.episode_id = EpisodeId(rollout_id="forged", attempt=1)
    elif field == "attempt":
        prefix.episode_id = EpisodeId(rollout_id="test", attempt=2)
    elif field == "task":
        prefix.task_id = TaskId(taskset="forged", task_id="other")
    elif field == "task_data":
        prefix.task_data["original_metadata"] = "forged"
    elif field == "params":
        prefix.responses_create_params.temperature = 0.25
    # Make the immutable execution identity internally consistent too; only the seed proves it is wrong.
    step = json.loads((root / "native-step.json").read_text())
    step.update(
        episode_id=prefix.episode_id.model_dump(),
        task_id=prefix.task_id.model_dump(),
        responses_create_params=prefix.model_dump(mode="json")["responses_create_params"],
    )
    (root / "native-step.json").write_text(json.dumps(step))
    prefix.response.replay_verification_sha256 = verification_digest(root)
    with pytest.raises(HTTPException, match="durable typed Resources seed"):
        await resource.prefix_review(
            prefix, request(session="unseeded" if field == "unseeded_cookie" else "resources-session")
        )
    assert not grades
    assert original.episode_id.rollout_id == "test"


def test_replay_opt_in_passes_public_resources_lifecycle_conformance(tmp_path, monkeypatch):
    resource, _ = server(tmp_path, monkeypatch)
    seed = ResourcesSeedSessionRequest(
        resources_session_id="lifecycle",
        episode_id=EpisodeId(rollout_id="lifecycle", attempt=1),
        task_id=TaskId(taskset="fixture", task_id="pif_029"),
        task_data={key: value for key, value in TASK.items() if key != "responses_create_params"},
        responses_create_params=PARAMS,
    )
    check_resources_session_contract(resource.setup_webserver(), seed, keeps_state=True)


@pytest.mark.asyncio
async def test_close_racing_review_waits_then_fences_new_reviews_and_retains_terminal_receipt(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, _ = server(tmp_path, monkeypatch)
    prefix = review_request(body)
    entered, finish = asyncio.Event(), asyncio.Event()
    original_grade = app.grade

    async def blocked_grade(*args, **kwargs):
        entered.set()
        await finish.wait()
        return await original_grade(*args, **kwargs)

    monkeypatch.setattr(app, "grade", blocked_grade)
    running = asyncio.create_task(seeded_review(resource, prefix, request()))
    await entered.wait()
    closing = asyncio.create_task(
        resource.close_resources_session(
            ResourcesCloseSessionRequest(resources_session_id="resources-session", episode_id=prefix.episode_id)
        )
    )
    await asyncio.sleep(0.02)
    assert not closing.done()
    finish.set()
    result = await running
    await closing
    with pytest.raises(HTTPException, match="already closed"):
        await resource.prefix_review(prefix, request())
    # Closing Resources does not erase the immutable receipt or permit another judge dispatch.
    final = await final_verify(resource, prefix, result)
    assert not final.mask_sample and final.response == result.response


@pytest.mark.asyncio
async def test_close_fence_expires_at_configured_window_and_survives_restart(tmp_path, monkeypatch):
    resource, _ = server(tmp_path, monkeypatch)
    resource.config.session_close_retry_window_seconds = 5
    instant = [100.0]
    monkeypatch.setattr(app, "time", lambda: instant[0])
    seed = ResourcesSeedSessionRequest(
        resources_session_id="fenced",
        episode_id=EpisodeId(rollout_id="test", attempt=1),
        task_id=TaskId(taskset="fixture", task_id="pif_029"),
        task_data={},
        responses_create_params=PARAMS,
    )
    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="fenced", episode_id=seed.episode_id)
    )
    restarted = app.PartnerIFResourcesServer(config=resource.config, server_client=resource.server_client)
    with pytest.raises(HTTPException, match="already closed"):
        await restarted.seed_session(seed, request=request())
    instant[0] = 106.0
    await restarted.seed_session(seed, request=request())


@pytest.mark.asyncio
async def test_new_close_owner_after_expired_fence_remains_idempotent(tmp_path, monkeypatch):
    resource, _ = server(tmp_path, monkeypatch)
    resource.config.session_close_retry_window_seconds = 5
    instant = [100.0]
    monkeypatch.setattr(app, "time", lambda: instant[0])
    first = EpisodeId(rollout_id="first", attempt=1)
    second = EpisodeId(rollout_id="second", attempt=1)
    await resource.seed_session(
        ResourcesSeedSessionRequest(
            resources_session_id="reused",
            episode_id=first,
            task_id=TaskId(taskset="fixture", task_id="pif_029"),
            task_data={},
            responses_create_params=PARAMS,
        ),
        request=request(),
    )
    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="reused", episode_id=first)
    )
    instant[0] = 106.0
    closing = ResourcesCloseSessionRequest(resources_session_id="reused", episode_id=second)
    assert await resource.close_resources_session(closing) == await resource.close_resources_session(closing)
    with pytest.raises(HTTPException, match="another episode"):
        await resource.close_resources_session(
            ResourcesCloseSessionRequest(resources_session_id="reused", episode_id=first)
        )


@pytest.mark.asyncio
async def test_disabled_replay_never_falls_back_to_judge_for_retained_review_failure(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    body.verification_input = app.LiveAgentCapture.model_validate(body.response.verification_input)
    body.response.replay_review_error = {"exception_type": "RuntimeError", "detail": "HTTP 422"}
    resource, grades = server(tmp_path, monkeypatch, opted_in=False)
    result = await resource.verify(body, request())
    assert not grades and result.mask_sample and not result.evaluation_completed
    assert result.failure_kind == "partner_if:replay_invalid_receipt"
