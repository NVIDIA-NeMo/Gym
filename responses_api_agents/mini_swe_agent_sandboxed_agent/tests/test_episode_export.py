# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-facing prompt boundaries, immutable request capture and exported API paths."""

import asyncio
import json
from copy import deepcopy

import pytest
from minisweagent.exceptions import FormatError
from pydantic import TypeAdapter

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.mini_swe_agent_sandboxed_agent.app import BASH_TOOL, SESSION_ID_KEY, SUBMIT_MARKER
from responses_api_agents.mini_swe_agent_sandboxed_agent.episode_export import (
    SCHEMA,
    canonical_input,
    episode_export,
    snapshot_hash,
)
from responses_api_agents.mini_swe_agent_sandboxed_agent.tests.test_reference import (
    FakeClient,
    FakeSandbox,
    _bash,
    _build,
    _message,
    _ok,
    _reasoning,
)
from responses_api_agents.mini_swe_agent_sandboxed_agent.tests.test_tb4 import build


INITIAL = [
    {"role": "system", "content": "system"},
    {"role": "developer", "content": "developer"},
    {"role": "user", "content": "example"},
    {"role": "assistant", "content": "few-shot answer"},
    {"role": "user", "content": [{"type": "input_text", "text": "actual task"}]},
]


@pytest.mark.parametrize("count", [0, 1, 2, 5])
@pytest.mark.parametrize("first", ["reasoning", "tool", "assistant"])
async def test_dynamic_boundary_is_the_complete_first_dispatched_context(count, first):
    generation = {
        "reasoning": [_reasoning("think", 1), _bash("pwd", 1)],
        "tool": [_bash("pwd", 1)],
        "assistant": [_message("acting", 1), _bash("pwd", 1)],
    }[first]
    client = FakeClient([generation])
    model, _, _ = _build(client, FakeSandbox([]))
    initial = deepcopy(INITIAL[:count])
    await model.query(initial)
    raw_generation = model.responses[0].model_dump(mode="json")["output"]
    feedback = canonical_input([{"role": "user", "content": "later feedback"}])
    full = canonical_input(initial) + raw_generation + feedback
    request, output, metadata = episode_export(full, model.request_capture())
    assert request["input"] == canonical_input(client.requests[0]["input"])
    assert metadata["prompt_boundary"] == count and metadata["normalization_valid"]
    assert output == raw_generation + feedback
    assert output[-1]["role"] == "user"
    assert (
        request["tools"]
        == NeMoGymResponseCreateParamsNonStreaming(input=[], tools=[BASH_TOOL]).model_dump(mode="json")["tools"]
    )
    assert model.request_capture()["first_request"]["tools"] == client.requests[0]["tools"]
    assert metadata["schema"] == SCHEMA


async def test_first_request_snapshot_survives_mutation_later_turns_and_timeout_retry():
    client = FakeClient([[_bash("pwd", 1)], [_bash("pwd", 2)]])
    original = client.create_response
    attempted = []

    async def dispatch(**kwargs):
        attempted.append(deepcopy(kwargs))
        if len(attempted) == 1:
            await asyncio.sleep(1)
        return await original(**kwargs)

    client.create_response = dispatch
    model, _, _ = _build(client, FakeSandbox([]))
    model._call_timeout_s = 0.01
    model._max_attempts = 2
    initial = deepcopy(INITIAL)
    await model.query(initial)
    captured = deepcopy(model.request_capture())
    initial[-1]["content"][0]["text"] = "mutated"
    initial.append({"role": "user", "content": "later"})
    await model.query(initial)
    assert model.request_capture()["first_request"] == attempted[0]
    assert model.request_capture()["first_request"] == captured["first_request"]
    assert model.request_capture()["request_attempt_count"] == 3
    assert model.calls_gt_timeout == 1
    captured["first_request"]["input"].clear()
    assert len(model.request_capture()["first_request"]["input"]) == 5


async def test_failed_first_dispatch_keeps_observed_prompt_and_omits_transport_secrets():
    client = FakeClient([])

    async def fail(**kwargs):
        raise RuntimeError("model unavailable")

    client.create_response = fail
    model, _, _ = _build(client, FakeSandbox([]))
    model.model_kwargs = {"temperature": 0.6, "extra_headers": {"Authorization": "secret-credential"}}
    with pytest.raises(RuntimeError, match="unavailable"):
        await model.query(deepcopy(INITIAL))
    capture = model.request_capture()
    request, output, metadata = episode_export(canonical_input(INITIAL), capture)
    assert capture["state"] == "attempted_no_response"
    assert capture["request_attempt_count"] == 1 and capture["returned_response_count"] == 0
    assert metadata["normalization_valid"] and metadata["prompt_boundary"] == 5
    assert request["input"] == canonical_input(INITIAL) and output == []
    assert request["temperature"] == 0.6
    assert capture["omitted_request_parameters"] == ["extra_headers"]
    assert "secret-credential" not in json.dumps(model.serialize())


def test_no_model_call_does_not_invent_a_request_or_generation():
    model, _, _ = _build(FakeClient([]), FakeSandbox([]))
    request, output, metadata = episode_export(canonical_input(INITIAL), model.request_capture())
    assert request["input"] == [] and request["tools"] == [] and output == []
    assert metadata["state"] == "no_model_call"
    assert metadata["prompt_boundary"] is None and not metadata["normalization_valid"]
    assert model.request_capture()["first_request"] is None
    assert snapshot_hash(None) is None


async def test_empty_first_response_uses_capture_and_keeps_later_feedback():
    model, _, _ = _build(FakeClient([[]]), FakeSandbox([]))
    with pytest.raises(FormatError):
        await model.query(deepcopy(INITIAL))
    feedback = canonical_input([{"role": "user", "content": "format error"}])
    request, output, metadata = episode_export(canonical_input(INITIAL) + feedback, model.request_capture())
    assert metadata["state"] == "first_response_empty" and metadata["normalization_valid"]
    assert request["input"] == canonical_input(INITIAL) and output == feedback


async def test_mismatched_or_interrupted_prefix_is_explicitly_invalid():
    model, _, _ = _build(FakeClient([[_bash("pwd", 1)]]), FakeSandbox([]))
    await model.query(deepcopy(INITIAL))
    for full in ([], canonical_input(INITIAL[:-1]), canonical_input([{"role": "user", "content": "wrong"}])):
        request, output, metadata = episode_export(full, model.request_capture())
        assert request["input"] == [] and output == []
        assert metadata["state"] == "boundary_error" and not metadata["normalization_valid"]
        assert metadata["error"]


async def test_format_error_recovery_preserves_audit_history_without_changing_model_replay():
    failed = _message("No tool call", 1)
    client = FakeClient([[failed], [_bash("echo " + SUBMIT_MARKER, 2)]])
    model, env, agent = _build(client, FakeSandbox([(SUBMIT_MARKER, _ok(SUBMIT_MARKER + "\n"))]))
    await env.prepare()
    await agent.run("synthetic task")
    request, output, metadata = episode_export(agent.trajectory, model.request_capture())
    exported = [item.model_dump(mode="json") for item in output]
    assert exported[0]["id"] == "msg_1" and exported[1]["role"] == "user"
    assert all(item.get("id") != "msg_1" for item in client.requests[1]["input"])
    assert metadata["flattened_history_is_exact_context_replay"] is False
    assert request["input"] == canonical_input(client.requests[0]["input"])


async def test_run_seed_verify_and_trajectory_reload_have_the_correct_boundary(tmp_path, monkeypatch):
    agent, request, body, _, _, _, _, _ = build(tmp_path, monkeypatch)
    original_body = body.model_dump(mode="json")
    post = agent.server_client.post
    transmitted = {}

    async def record(**kwargs):
        transmitted[kwargs["url_path"]] = deepcopy(kwargs["json"])
        return await post(**kwargs)

    agent.server_client.post = record
    result = await agent.run(request, body)
    row = result.model_dump(mode="json")
    assert body.model_dump(mode="json") == original_body
    assert (
        transmitted["/seed_session"]["responses_create_params"]["input"]
        == original_body["responses_create_params"]["input"]
    )
    assert transmitted["/verify"]["responses_create_params"] == row["responses_create_params"]
    assert row["responses_create_params"]["input"] == canonical_input(
        row["mini_swe_model_request_capture"]["first_request"]["input"]
    )
    assert row["response"]["output"][0]["type"] == "reasoning"
    tool_list = TypeAdapter(NeMoGymResponse.model_fields["tools"].annotation)
    assert tool_list.dump_python(
        tool_list.validate_python(row["response"]["tools"]), mode="json"
    ) == tool_list.dump_python(tool_list.validate_python(row["responses_create_params"]["tools"]), mode="json")
    assert row["responses_create_params"]["tools"][0]["name"] == "bash"
    from pathlib import Path

    archive = json.loads(Path(row["mini_swe_trajectory_path"]).read_text())
    assert archive["responses_create_params"] == row["responses_create_params"]
    assert archive["model_request_capture"] == row["mini_swe_model_request_capture"]
    boundary = row["mini_swe_export"]["prompt_boundary"]
    assert archive["gym_full_trajectory"][boundary:] == row["response"]["output"]
    assert row["reward"] == 1 and row["evaluation_completed"]
    assert row["response"]["usage"]["output_tokens"] == 10
    assert archive["shell_records"] == row["mini_swe_shell_records"]


async def test_bare_responses_round_trip_keeps_export_request_extension(tmp_path, monkeypatch):
    agent, request, body, _, sandbox, _, _, _ = build(tmp_path, monkeypatch)
    request.session[SESSION_ID_KEY] = "bare"
    agent._session_sandboxes["bare"] = sandbox
    response = await agent.responses(request, body.responses_create_params)
    row = NeMoGymResponse.model_validate(response.model_dump(mode="json")).model_dump(mode="json")
    assert row["mini_swe_export"]["normalization_valid"]
    assert row["output"][0]["type"] == "reasoning"
    assert row["responses_create_params"]["tools"][0]["name"] == "bash"
