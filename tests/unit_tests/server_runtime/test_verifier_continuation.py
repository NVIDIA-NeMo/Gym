# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted verifier requests use bounded fresh sessions and preserve the episode's evidence."""

import asyncio

import orjson
import pytest
from pydantic import ValidationError

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    VerifierContinuationConfig,
    _continuation_evidence,
    _ContinuationEvidenceScope,
)
from nemo_gym.core.responses_api_types import NeMoGymResponseUsage
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnResult
from nemo_gym.trajectory.model_call_record import ModelCallRecord
from nemo_gym.trajectory.trajectory_record import (
    AgentObservationBundle,
    TrajectoryRecord,
    grading_result_binding,
    join_agent_model_calls,
)
from tests.helpers.single_agent_turn import _agent_response, _Client, _environment_server, _request, _Response


NEXT = {"input": "Continue the original task", "metadata": {"next_step": "2"}}


class _ContinuationClient(_Client):
    sign_grades: bool = False

    async def post(self, server_name, url_path, **kwargs):
        reply = await super().post(server_name, url_path, **kwargs)
        if url_path == "/verify":
            payload = orjson.loads(reply.body)
            payload["response"]["usage"] = kwargs["json"]["response"]["usage"]
            if self.sign_grades:
                payload["ng_verifier_invocations"][0]["grading_record"] = {
                    "verifier": "resources",
                    "score": payload["reward"],
                    "verdict": "masked" if payload["mask_sample"] else "scored",
                    "result_binding_sha256": grading_result_binding(payload),
                }
            reply = _Response(payload)
        return reply


def continuation_server(*, max_activations=3, next_request=NEXT, masked_intermediate=True, cumulative=False):
    server, client = _environment_server(client_type=_ContinuationClient)
    server.config.verifier_continuation = VerifierContinuationConfig(max_activations=max_activations)
    replies = [_Response({"resources_session_id": "resources"})]
    earlier = []
    for index in (1, 2):
        response = _agent_response().model_copy(
            update={
                "id": f"response-{index}",
                "usage": NeMoGymResponseUsage(
                    input_tokens=2,
                    output_tokens=1,
                    total_tokens=3,
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": 0},
                ),
            }
        )
        invocation = {"kind": "agent_invocation", "invocation_id": f"actor-{index}", "status": "completed"}
        records = [*earlier, invocation] if cumulative else [invocation]
        trajectory = {
            "task_id": "task",
            "rollout_id": "rollout-a2",
            "invocations": records,
            "turns": [
                {
                    "invocation_id": record["invocation_id"],
                    "task_id": "task",
                    "rollout_id": "rollout-a2",
                    "turn_no": 1,
                    "timestamp": position,
                    "step_count": 1,
                }
                for position, record in enumerate(records, 1)
            ],
        }
        replies.extend(
            [
                _Response({"agent_session_id": "agent"}),
                _Response(response.model_dump(mode="json")),
                _Response(
                    {
                        "agent_session_id": "agent",
                        "cleanup_confirmed": True,
                        "agent_observations": {
                            "source": "synthetic",
                            "records": records,
                            "gaps": [{"code": "unavailable", "detail": "No native timing"}],
                        },
                        "trajectory": trajectory,
                    }
                ),
                _Response(
                    {
                        "responses_create_params": {"input": "task" if index == 1 else NEXT["input"]},
                        "response": response.model_dump(mode="json"),
                        "reward": 0.0 if index == 1 else 1.0,
                        "mask_sample": masked_intermediate if index == 1 else False,
                        "next_agent_request": next_request if index == 1 else None,
                        "ng_verifier_invocations": [{"invocation_id": f"verify-{index}", "status": "completed"}],
                    }
                ),
            ]
        )
        earlier.append(invocation)
    replies.append(_Response({"resources_session_id": "resources"}))
    client.responses = replies
    return server, client


@pytest.mark.asyncio
@pytest.mark.parametrize("masked_intermediate", [False, True])
async def test_verifier_controls_fresh_activations_without_changing_task_identity(masked_intermediate):
    server, client = continuation_server(masked_intermediate=masked_intermediate)
    request = _request()
    before = request.model_dump(mode="json")

    result = await server.run_request(request)

    assert result.failure is None
    assert result.result.reward == 1.0 and not result.result.mask_sample
    paths = [path for _, path, _ in client.calls]
    assert paths == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    seeds = [call["json"] for _, path, call in client.calls if path == "/v1/agent_sessions"]
    assert len({seed["agent_session_id"] for seed in seeds}) == 2
    assert seeds[0]["task_id"] == seeds[1]["task_id"]
    activations = [call for _, path, call in client.calls if path.endswith("/v1/responses")]
    assert [call["headers"]["x-ng-agent-activation"] for call in activations] == ["1", "1"]
    assert activations[1]["json"] == NEXT
    assert request.model_dump(mode="json") == before
    assert "next_agent_request" not in result.result.model_dump(mode="json")
    assert result.result.responses_create_params.input == "task"
    assert result.result.agent_activations == result.result.verifier_calls == 2
    assert result.result.response.usage.total_tokens == 6
    assert [record.invocation_id.rsplit(":", 1)[-1] for record in result.result.ng_agent_observations.records] == [
        "actor-1",
        "actor-2",
    ]
    assert len(result.result.ng_agent_observations.gaps) == 1
    assert len(result.result.ng_trajectory.turns) == 2
    assert [record.invocation_id for record in result.result.ng_verifier_invocations] == ["verify-1", "verify-2"]
    assert len(result.result.ng_trajectory.verifier_invocations) == 2


@pytest.mark.asyncio
async def test_exact_repeated_evidence_is_retained_once():
    server, _ = continuation_server(cumulative=True)
    result = await server.run_request(_request())
    assert result.failure is None
    assert len(result.result.ng_agent_observations.records) == 2
    assert len(result.result.ng_trajectory.invocations) == 2
    assert len(result.result.ng_trajectory.turns) == 2


@pytest.mark.asyncio
async def test_fresh_sessions_scope_local_root_and_child_owners_without_cross_joining_captures():
    server, client = continuation_server()
    for index, close_index in enumerate((3, 7), 1):
        close = orjson.loads(client.responses[close_index].body)
        root = {
            "kind": "agent_invocation",
            "invocation_id": "root",
            "status": "completed",
            "model_calls": [{"model_call_id": f"call-{index}"}],
        }
        child = {
            "kind": "agent_invocation",
            "invocation_id": "child",
            "parent_invocation_id": "root",
            "spawned_by_tool_call_id": "spawn",
            "harness_session_id": "native-session",
            "status": "completed",
        }
        tool = {
            "kind": "tool_call",
            "invocation_id": "child",
            "tool_call_id": "native-tool",
            "status": "completed",
        }
        gap = {"code": "child-gap", "invocation_id": "child"}
        close["agent_observations"]["records"] = [
            root,
            child,
            tool,
            {"kind": "context_compaction", "invocation_id": "child", "summary": f"round {index}"},
        ]
        close["agent_observations"]["gaps"] = [gap]
        trajectory = close["trajectory"]
        trajectory["invocations"] = [root, child]
        trajectory["turns"][0].update(invocation_id="root", timestamp=index)
        trajectory["tool_calls"] = [tool]
        trajectory["model_calls"] = [{"model_call_id": f"call-{index}", "client_session_id": "root"}]
        trajectory["gaps"] = [gap]
        client.responses[close_index] = _Response(close)

    result = await server.run_request(_request())
    assert result.failure is None
    seeds = [call["json"] for _, path, call in client.calls if path == "/v1/agent_sessions"]
    expected_roots = [f"{seed['agent_session_id']}:root" for seed in seeds]
    expected_children = [f"{seed['agent_session_id']}:child" for seed in seeds]
    trajectory = result.result.ng_trajectory
    assert [turn.invocation_id for turn in trajectory.turns] == expected_roots
    assert [tool.invocation_id for tool in trajectory.tool_calls] == expected_children
    assert [gap.invocation_id for gap in trajectory.gaps] == expected_children
    assert [call.client_session_id for call in trajectory.model_calls] == expected_roots
    assert [call.model_call_id for call in trajectory.model_calls] == ["call-1", "call-2"]
    children = [inv for inv in trajectory.invocations if inv.parent_invocation_id is not None]
    assert [inv.parent_invocation_id for inv in children] == expected_roots
    assert all(
        inv.spawned_by_tool_call_id == "spawn" and inv.harness_session_id == "native-session" for inv in children
    )
    assert all(tool.tool_call_id == "native-tool" for tool in trajectory.tool_calls)
    bundle = result.result.ng_agent_observations
    assert [item.invocation_id for item in bundle.records if item.kind == "context_compaction"] == expected_children
    captured = [
        ModelCallRecord(model_call_id=f"call-{index}", client_session_id="root", call_index=index) for index in (1, 2)
    ]
    joined = join_agent_model_calls(bundle, captured)
    roots = [item for item in joined.records if item.kind == "agent_invocation" and item.parent_invocation_id is None]
    assert [[call.model_call_id for call in item.model_calls] for item in roots] == [["call-1"], ["call-2"]]
    assert not any(
        gap.code in {"model_call_ownership_conflict", "model_call_ownership_unavailable"} for gap in joined.gaps
    )


def test_continuation_rejects_conflicting_canonical_v2_spans_without_rewriting_native_identity():
    importer = _ContinuationEvidenceScope()
    trajectories = []
    for index in (1, 2):
        trajectory = TrajectoryRecord.model_validate(
            {
                "schema_version": "2.0",
                "task_id": "task",
                "capture_key": "rollout-a2",
                "invocations": [{"invocation_id": "root", "status": "completed", "duration_ms": index}],
                "tool_calls": [
                    {"invocation_id": "root", "tool_call_id": "native", "span_id": "canonical", "status": "completed"}
                ],
            }
        )
        bundle = AgentObservationBundle(schema_version="2.0", source="test", records=trajectory.invocations)
        _, imported = importer.import_session(f"session-{index}", bundle, trajectory)
        assert imported.tool_calls[0].span_id == "canonical"
        trajectories.append(imported)
    with pytest.raises(ValueError, match="conflicting canonical tool spans"):
        _continuation_evidence(_request(), [], trajectories, [])


@pytest.mark.asyncio
async def test_final_grade_binds_the_exact_usage_aggregated_response_returned_to_the_caller():
    server, client = continuation_server()
    client.sign_grades = True
    result = await server.run_request(_request())
    assert result.failure is None
    verified = [kwargs["json"]["response"] for _, path, kwargs in client.calls if path == "/verify"]
    assert [response["usage"]["total_tokens"] for response in verified] == [3, 6]
    assert result.result.response.model_dump(mode="json") == verified[-1]
    latest = result.result.ng_verifier_invocations[-1].grading_record
    assert latest.result_binding_sha256 == grading_result_binding(result.result)
    assert result.result.ng_trajectory.verifier_invocations[-1].grading_record == latest


def test_already_session_qualified_native_owners_retain_their_identity():
    bundle = AgentObservationBundle.model_validate(
        {
            "source": "native",
            "records": [
                {"kind": "agent_invocation", "invocation_id": "session-1"},
                {
                    "kind": "agent_invocation",
                    "invocation_id": "session-1:delegate",
                    "parent_invocation_id": "session-1",
                },
            ],
        }
    )
    imported, _ = _ContinuationEvidenceScope().import_session("session-1", bundle, None)
    assert imported == bundle


@pytest.mark.asyncio
@pytest.mark.parametrize("producer_trajectory", [False, True])
async def test_cap_discards_provisional_grade_claims_and_preserves_bound_prefix_receipts_as_history(
    producer_trajectory,
):
    server, client = continuation_server(max_activations=1, masked_intermediate=False)
    client.sign_grades = True
    provisional = orjson.loads(client.responses[4].body)
    provisional.update(
        reward=1.0, reward_components={"rubric": 1.0}, grading_result={"status": "scored"}, provisional_only="discard"
    )
    close = orjson.loads(client.responses[3].body)
    if not producer_trajectory:
        close.pop("trajectory")
        close.pop("agent_observations")
    client.responses = [*client.responses[:3], _Response(close), _Response(provisional), client.responses[-1]]
    request = _request()
    request.task.task_input.task_data["benchmark_metadata"] = "preserved"
    result = await server.run_request(request)
    assert result.failure is None
    assert result.result.mask_sample and not result.result.evaluation_completed
    assert result.result.reward == 0 and result.result.response.status == "incomplete"
    assert result.result.benchmark_metadata == "preserved"
    record = result.result.model_dump(mode="json")
    assert not {"reward_components", "grading_result", "provisional_only", "ng_verifier_invocations"} & record.keys()
    historical = result.result.ng_trajectory.verifier_invocations[0].grading_record
    assert historical.score == 1.0 and historical.verdict == "scored"
    verified_response = next(kwargs["json"]["response"] for _, path, kwargs in client.calls if path == "/verify")
    assert historical.result_binding_sha256 == grading_result_binding(
        {"response": verified_response, "reward": 1.0, "reward_components": {"rubric": 1.0}}
    )
    assert historical.result_binding_sha256 != grading_result_binding(result.result)


@pytest.mark.asyncio
async def test_a_continuation_request_at_the_limit_is_masked_without_another_model_call():
    server, client = continuation_server(max_activations=1)
    client.responses = [*client.responses[:5], client.responses[-1]]
    result = await server.run_request(_request())
    assert result.failure is None
    assert result.result.mask_sample and result.result.reward == 0.0
    assert result.result.failure_kind == "single_agent_turn:continuation_limit"
    assert result.result.evaluation_completed is False
    assert result.result.response.status == "incomplete"
    assert "next_agent_request" not in result.result.model_dump(mode="json")
    assert len([path for _, path, _ in client.calls if path.endswith("/v1/responses")]) == 1
    assert len(result.result.ng_trajectory.turns) == 1


@pytest.mark.asyncio
async def test_terminal_masked_result_does_not_continue():
    server, client = continuation_server(next_request=None)
    client.responses = [*client.responses[:5], client.responses[-1]]
    result = await server.run_request(_request())
    assert result.failure is None
    assert result.result.mask_sample and result.result.reward == 0.0
    assert len([path for _, path, _ in client.calls if path == "/verify"]) == 1


@pytest.mark.asyncio
async def test_missing_usage_remains_unknown_after_multiple_activations():
    server, client = continuation_server()
    response = orjson.loads(client.responses[6].body)
    response["usage"] = None
    client.responses[6] = _Response(response)
    result = await server.run_request(_request())
    assert result.failure is None
    assert result.result.response.usage is None


@pytest.mark.asyncio
async def test_unconfirmed_cleanup_prevents_verification_and_preserves_partial_evidence():
    server, client = continuation_server()
    close = orjson.loads(client.responses[3].body) | {
        "cleanup_confirmed": False,
        "resources_cookies": {"session": "after-unconfirmed-close"},
    }
    client.responses = [
        *client.responses[:3],
        _Response(close),
        _Response(close | {"cleanup_confirmed": True}),
        client.responses[-1],
    ]
    result = await server.run_request(_request())
    assert result.failure is not None and result.failure.stage == "cleanup"
    assert result.failure.partial_response.id == "response-1"
    assert result.failure.agent_observations.records[0].invocation_id.endswith(":actor-1")
    assert not any(path == "/verify" for _, path, _ in client.calls)
    assert client.calls[-1][1] == "/close_session"
    assert client.calls[-1][2]["cookies"] == {"session": "after-unconfirmed-close"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_round_restarts_and_closes_without_resending_the_model_request(monkeypatch, cancel):
    server, client = continuation_server()
    client.responses = [client.responses[0], client.responses[1], client.responses[3], client.responses[-1]]
    events = []
    original_post = _Client.post

    async def restarted(self, request):
        assert not client.calls
        events.append("restart")

    async def post(self, server_name, url_path, **kwargs):
        if url_path.endswith("/v1/responses"):
            self.calls.append((server_name, url_path, kwargs))
            events.append("model")
            raise asyncio.CancelledError() if cancel else TimeoutError("lost activation reply")
        return await original_post(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(SingleAgentTurnEnvironmentServer, "checkpoint_restart", restarted)
    monkeypatch.setattr(_Client, "post", post)
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await server.run_request(_request())
    else:
        result = await server.run_request(_request())
        assert result.failure is not None and result.failure.stage == "agent"
        assert result.failure.agent_observations.records[0].invocation_id.endswith(":actor-1")
    assert events == ["restart", "model"]
    assert client.calls[-2][1] == "/v1/agent_sessions/close"
    assert client.calls[-1][1] == "/close_session"
    assert not any(path == "/verify" for _, path, _ in client.calls)


@pytest.mark.asyncio
async def test_verifier_continuation_cannot_silently_skip_verification():
    server, client = continuation_server()
    server.config.skip_verification = True
    result = await server.run_request(_request())
    assert result.failure is not None and result.failure.terminal
    assert not client.calls


@pytest.mark.parametrize("limit", [0, 33])
def test_continuation_limit_is_bounded(limit):
    with pytest.raises(ValidationError):
        VerifierContinuationConfig(max_activations=limit)


def test_result_directive_uses_standard_typed_request_and_is_omitted_when_absent():
    result = SingleAgentTurnResult.model_validate(
        {
            "responses_create_params": {"input": "task"},
            "response": _agent_response(),
            "reward": 0,
            "next_agent_request": NEXT,
        }
    )
    assert result.next_agent_request.metadata == NEXT["metadata"]
    result.next_agent_request = None
    assert "next_agent_request" not in result.model_dump(mode="json")
    malformed = result.model_dump(mode="json") | {"next_agent_request": {"input": "task", "invented_option": True}}
    with pytest.raises(ValidationError):
        SingleAgentTurnResult.model_validate(malformed)


@pytest.mark.parametrize("explicit", [None, False, True], ids=["omitted", "explicit-false", "explicit-true"])
def test_typed_directive_json_round_trip_preserves_only_explicit_actor_parameters(explicit):
    params = {"input": "Continue the original task", "metadata": {"next_step": "2"}}
    if explicit is not None:
        params["parallel_tool_calls"] = explicit
    result = SingleAgentTurnResult.model_validate(
        {
            "responses_create_params": {"input": "task"},
            "response": _agent_response(),
            "reward": 0,
            "next_agent_request": params,
        }
    )

    assert result.model_dump(mode="json")["next_agent_request"] == params
    wire = result.model_dump_json()
    assert orjson.loads(wire)["next_agent_request"] == params
    restored = SingleAgentTurnResult.model_validate_json(wire)
    assert restored.next_agent_request.model_dump(mode="json", exclude_none=True, exclude_unset=True) == params


@pytest.mark.asyncio
async def test_reply_trajectory_is_retained_when_close_has_observations_only():
    server, client = continuation_server()
    for reply_index, close_index in ((2, 3), (6, 7)):
        reply = orjson.loads(client.responses[reply_index].body)
        close = orjson.loads(client.responses[close_index].body)
        reply["_ng_trajectory"] = close.pop("trajectory")
        client.responses[reply_index] = _Response(reply)
        client.responses[close_index] = _Response(close)
    result = await server.run_request(_request())
    assert result.failure is None
    assert len(result.result.ng_trajectory.turns) == 2


@pytest.mark.asyncio
async def test_final_verifier_response_status_and_metadata_are_retained():
    server, client = continuation_server(next_request=None)
    verified = orjson.loads(client.responses[4].body)
    verified["response"].update(status="incomplete", metadata={"diagnostic": "undecided"})
    client.responses = [*client.responses[:4], _Response(verified), client.responses[-1]]
    result = await server.run_request(_request())
    assert result.failure is None
    assert result.result.response.status == "incomplete"
    assert result.result.response.metadata == {"diagnostic": "undecided"}
