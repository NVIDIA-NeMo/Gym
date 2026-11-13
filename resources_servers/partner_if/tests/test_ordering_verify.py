# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native capture round trips; semantic verdicts are controlled fixtures."""

import copy
import json
from http.cookies import SimpleCookie
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from benchmarks.partner_if.tests.fixture_data import BUNDLE, PARAMS, PROBE, SOURCE, TASK
from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.partner_if import app
from resources_servers.partner_if.engine import aggregate
from resources_servers.partner_if.grading_contract import RawJudgment
from resources_servers.partner_if.grading_service import GradeResponse
from responses_api_agents.ngc_recorded_context import native_agent
from responses_api_agents.ngc_recorded_context.replay import ReplayConfig, actor_identity
from responses_api_agents.ngc_recorded_context.saved_prefix import verification_digest


def test_next_activation_serialization_keeps_the_typed_request_schema():
    schema = app.VerifyResponse.model_json_schema(mode="serialization")
    variants = schema["properties"]["next_agent_request"]["anyOf"]
    assert {"$ref": "#/$defs/NeMoGymResponseCreateParamsNonStreaming"} in variants


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
    state = await agent._seed_agent_session_state(seed)
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


def server(tmp_path, monkeypatch, *, passed=True, opted_in=True, max_calls=8):
    calls = []

    async def semantic_fixture(contract, evidence, **kwargs):
        calls.append(evidence)
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
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    return result, calls


@pytest.mark.asyncio
async def test_safe_prefix_returns_masked_typed_request_without_changing_actor_inputs(tmp_path, monkeypatch):
    body, root = await native_capture(
        tmp_path, monkeypatch, calls=[call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})]
    )
    resource, grades = server(tmp_path, monkeypatch)
    result = await resource.verify(body)
    assert len(grades) == 1
    assert actor_identity({"input": grades[0].context}) == actor_identity({"input": PARAMS["input"]})
    assert result.mask_sample and not result.evaluation_completed and result.reward == 0
    assert result.replay_diagnostics["termination"] == "ordering_pending"
    next_request = result.next_agent_request
    assert next_request.input == body.responses_create_params.input
    assert next_request.tools == body.responses_create_params.tools
    assert next_request.metadata == {
        "ngc_replay_parent_turn_limit": "2",
        "ngc_replay_saved_prefix": str(root),
        "ngc_replay_saved_prefix_sha256": body.response.saved_prefix_sha256,
    }
    assert "parent_turn_limit" not in next_request.model_dump()
    assert "next_agent_request" in result.model_dump(mode="json")


@pytest.mark.asyncio
async def test_continuation_json_roundtrip_preserves_omitted_actor_parameters(tmp_path, monkeypatch):
    body, _ = await native_capture(
        tmp_path, monkeypatch, calls=[call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})]
    )
    resource, _ = server(tmp_path, monkeypatch)
    result = await resource.verify(body)
    assert "parallel_tool_calls" not in body.responses_create_params.model_fields_set
    wire = json.loads(result.model_dump_json())["next_agent_request"]
    assert "parallel_tool_calls" not in wire
    restored = app.VerifyResponse.model_validate_json(result.model_dump_json())
    assert restored.next_agent_request.model_fields_set == result.next_agent_request.model_fields_set
    assert "parallel_tool_calls" not in restored.next_agent_request.model_fields_set
    actor = restored.next_agent_request.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    metadata = actor.pop("metadata")
    assert set(metadata) == {
        "ngc_replay_parent_turn_limit",
        "ngc_replay_saved_prefix",
        "ngc_replay_saved_prefix_sha256",
    }
    original = body.responses_create_params.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    assert actor_identity(actor) == actor_identity(original)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passed,calls",
    [
        (True, [probe()]),
        (False, [call("Write", {"file_path": "/workspace/logs/answer.txt", "content": "Publish a replacement now."})]),
        (
            False,
            [probe(), call("Write", {"file_path": "/workspace/logs/answer.txt", "content": "Publish now."}, "draft")],
        ),
    ],
)
async def test_decisive_prefix_retains_shared_verdict_and_never_continues(tmp_path, monkeypatch, passed, calls):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=calls)
    resource, grades = server(tmp_path, monkeypatch, passed=passed)
    result = await resource.verify(body)
    assert len(grades) == 1
    assert result.reward == float(passed) and not result.mask_sample and result.evaluation_completed
    assert result.replay_diagnostics["termination"] == ("ordering_passed" if passed else "ordering_failed")
    assert result.next_agent_request is None
    assert "next_agent_request" not in result.model_dump(mode="json")


@pytest.mark.asyncio
async def test_safe_final_without_probe_is_masked_not_a_false_pass(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, text="Still investigating.")
    resource, grades = server(tmp_path, monkeypatch)
    result = await resource.verify(body)
    assert len(grades) == 1 and result.mask_sample and not result.evaluation_completed
    assert result.replay_diagnostics["termination"] == "ordering_undecided"
    assert result.next_agent_request is None


@pytest.mark.asyncio
async def test_technical_failure_is_masked_without_judge_or_continuation(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, error=TimeoutError("scripted provider fault"))
    resource, grades = server(tmp_path, monkeypatch)
    result = await resource.verify(body)
    assert not grades and result.mask_sample and result.next_agent_request is None
    assert result.replay_diagnostics["termination"] == "provider_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("decisive", [False, True])
async def test_policy_budget_masks_pending_but_preserves_decisive_last_call(tmp_path, monkeypatch, decisive):
    calls = [probe()] if decisive else [call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})]
    body, _ = await native_capture(tmp_path, monkeypatch, calls=calls)
    resource, _ = server(tmp_path, monkeypatch, max_calls=1)
    result = await resource.verify(body)
    assert result.mask_sample is not decisive and result.next_agent_request is None
    assert result.replay_diagnostics["termination"] == ("ordering_passed" if decisive else "policy_call_limit")


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
        body.verification_input = copy.deepcopy(body.response.verification_input)
        body.verification_input = app.LiveAgentCapture.model_validate(body.verification_input)
        body.verification_input.input = []
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
    result = await resource.verify(body)
    assert not grades and result.mask_sample and result.next_agent_request is None
    assert result.replay_diagnostics["termination"] == "invalid_capture"
    assert result.replay_artifacts_dir == body.response.replay_artifacts_dir


@pytest.mark.asyncio
async def test_opt_out_does_not_lift_nested_capture(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, grades = server(tmp_path, monkeypatch, opted_in=False)
    result = await resource.verify(body)
    assert not grades and result.mask_sample and result.next_agent_request is None
    assert "replay_diagnostics" not in result.model_dump()


@pytest.mark.asyncio
async def test_explicit_matching_capture_is_accepted_and_plain_live_agent_rows_are_unchanged(tmp_path, monkeypatch):
    body, _ = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    body.verification_input = app.LiveAgentCapture.model_validate(body.response.verification_input)
    resource, grades = server(tmp_path, monkeypatch)
    result = await resource.verify(body)
    assert result.reward == 1 and not result.mask_sample and len(grades) == 1
    # The same live-agent evidence without any replay envelope follows the ordinary path even on opted-in server.
    plain = body.model_copy(deep=True)
    for key in list(plain.response.model_extra):
        plain.response.model_extra.pop(key)
    result = await resource.verify(plain)
    assert result.reward == 1 and not result.mask_sample and result.next_agent_request is None
    assert "replay_diagnostics" not in result.model_dump()


@pytest.mark.asyncio
async def test_production_judge_failsafe_masks_outage_and_preserves_capture_without_continuation(
    tmp_path, monkeypatch
):
    body, root = await native_capture(tmp_path, monkeypatch, calls=[probe()])
    resource, _ = server(tmp_path, monkeypatch)

    async def failed_judge(*args, **kwargs):
        raise JudgeError("scripted judge unavailable")

    monkeypatch.setattr(app, "grade", failed_judge)
    result = await judge_failsafe(resource.verify)(body)
    record = json.loads(result.body)
    assert record["mask_sample"] is True and record["reward"] == 0
    assert record["_ng_failure_class"] == "judge_failed"
    assert record.get("next_agent_request") is None
    assert record["response"]["replay_artifacts_dir"] == str(root)


@pytest.mark.asyncio
async def test_continuation_replaces_previous_controls_and_keeps_other_metadata(tmp_path, monkeypatch):
    body, root = await native_capture(
        tmp_path, monkeypatch, calls=[call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})]
    )
    body.responses_create_params.metadata = {
        "origin": "unchanged",
        "ngc_replay_parent_turn_limit": "99",
        "ngc_replay_saved_prefix": "stale",
        "ngc_replay_saved_prefix_sha256": "0" * 64,
    }
    resource, _ = server(tmp_path, monkeypatch)
    result = await resource.verify(body)
    assert result.next_agent_request.metadata["origin"] == "unchanged"
    assert result.next_agent_request.metadata["ngc_replay_parent_turn_limit"] == "2"
    assert result.next_agent_request.metadata["ngc_replay_saved_prefix"] == str(root)
