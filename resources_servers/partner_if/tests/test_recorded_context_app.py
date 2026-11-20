# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources owns frozen evidence, budgets, idempotent judgment, and Gymnasium cleanup."""

import asyncio
import copy
import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from benchmarks.partner_if.tests.fixture_data import BUNDLE, PARAMS, PROBE, SOURCE, TASK
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import rollout_context
from nemo_gym.verification import JudgeError
from resources_servers.gymnasium import EnvResetRequest, EnvStepRequest
from resources_servers.partner_if import app
from resources_servers.partner_if.engine import aggregate
from resources_servers.partner_if.grading_contract import RawJudgment
from resources_servers.partner_if.grading_service import GradeResponse
from resources_servers.partner_if.recorded_context.assets import ReplayConfig
from resources_servers.partner_if.recorded_context_app import (
    RESULT_BINDING_KEY,
    RESULT_KEY,
    RecordedContextResourcesServer,
    RecordedContextResourcesServerConfig,
    recorded_context_result_sha256,
)


def request(session="episode"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/step",
            "headers": [],
            "session": {SESSION_ID_KEY: session},
            "app": FastAPI(),
        }
    )


def tool(name, arguments, call_id="call"):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def probe():
    return tool("Bash", {"command": f"python3 -I {PROBE} --uri '{BUNDLE['sibling']['input_uri']}'"}, "probe")


def read():
    return tool("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}, "read")


def response(*calls, text=None, identity="response"):
    return {
        "id": identity,
        "created_at": 0,
        "model": "test",
        "object": "response",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": (
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ]
            if text is not None
            else []
        )
        + list(calls),
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": None},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": None},
            "total_tokens": 120,
        },
    }


def server(monkeypatch, *, passed=True, max_parent_turns=6, observed=False):
    calls = []

    async def grade(contract, evidence, **kwargs):
        calls.append(copy.deepcopy(evidence))
        return GradeResponse(
            grading_result=aggregate(
                contract,
                [
                    RawJudgment(
                        id=criterion.id,
                        passed=passed,
                        reason="Scripted semantic verdict",
                        evidence="Public synthetic fixture",
                    )
                    for criterion in contract.checks
                ],
            )
        )

    monkeypatch.setattr(app, "grade", grade)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": observed}
    resource = RecordedContextResourcesServer(
        config=RecordedContextResourcesServerConfig(
            name="recorded",
            host="127.0.0.1",
            port=0,
            entrypoint="recorded_context_app.py",
            policy_model="test",
            policy_model_server={"type": "responses_api_models", "name": "policy"},
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            replay=ReplayConfig(package_dir=SOURCE / "assets", coverage_profile="expanded-v4"),
            max_parent_turns=max_parent_turns,
        ),
        server_client=client,
    )
    return resource, calls


async def reset(resource, row=None):
    return await resource._reset_endpoint(EnvResetRequest.model_validate(row or TASK), request())


async def step(resource, reply, *, identity="step", row=None):
    return await resource._step_endpoint(
        EnvStepRequest.model_validate((row or TASK) | {"response": reply, "_ng_step_request_id": identity}), request()
    )


@pytest.mark.asyncio
async def test_reset_has_no_actor_message_and_validates_the_frozen_task(monkeypatch):
    resource, calls = server(monkeypatch)
    initialized = await reset(resource)
    assert initialized.observation is None and initialized.info["supports_step_idempotency"]
    assert initialized.info["supports_explicit_close"] and not calls
    changed = copy.deepcopy(TASK)
    changed["responses_create_params"]["input"] = []
    with pytest.raises(HTTPException):
        await resource._reset_endpoint(EnvResetRequest.model_validate(changed), request("another"))


@pytest.mark.asyncio
async def test_step_retries_and_terminal_retries_do_not_repeat_tools_or_judge(monkeypatch):
    resource, calls = server(monkeypatch)
    await reset(resource)
    body = response(probe())
    results = await asyncio.gather(step(resource, body), step(resource, body))
    assert len(calls) == 1 and results[0] == results[1]
    exported = results[0].info[RESULT_KEY]
    assert exported[RESULT_BINDING_KEY] == recorded_context_result_sha256(exported)
    assert exported["response"]["output"][-1]["type"] == "function_call_output"
    assert exported["recorded_context"]["model_calls"] == 1
    await resource._close_endpoint(request())
    assert await step(resource, body) == results[0]
    assert len(calls) == 1
    with pytest.raises(HTTPException, match="different response"):
        await step(resource, response(probe(), identity="changed"))


@pytest.mark.asyncio
@pytest.mark.parametrize("output_option", ["--output ", "--output="])
async def test_step_can_read_a_probe_output_in_the_same_response(monkeypatch, output_option):
    resource, calls = server(monkeypatch)
    await reset(resource)
    path = "/workspace/logs/probe.json"
    command = f"python3 -I {PROBE} --uri '{BUNDLE['sibling']['input_uri']}' {output_option}{path}"
    reply = await step(
        resource,
        response(tool("Bash", {"command": command}, "probe"), tool("Read", {"file_path": path}, "read-probe")),
    )
    assert len(calls) == 1 and reply.terminated and not reply.mask_sample
    outputs = reply.info[RESULT_KEY]["response"]["output"]
    saved = next(
        item["output"]
        for item in outputs
        if item["type"] == "function_call_output" and item["call_id"] == "read-probe"
    )
    contents = "\n".join(line.partition("\t")[2] for line in saved.splitlines())
    assert json.loads(contents) == BUNDLE["sibling"]


@pytest.mark.asyncio
async def test_virtual_directory_collision_masks_the_step_instead_of_crashing(monkeypatch):
    resource, _ = server(monkeypatch)
    await reset(resource)
    path = "/workspace/logs/node"
    reply = await step(
        resource,
        response(
            tool("Write", {"file_path": path, "content": "first"}, "write-parent"),
            tool("Write", {"file_path": f"{path}/child.txt", "content": "needle"}, "write-child"),
            tool("Bash", {"command": "rg needle /workspace/logs"}, "search"),
        ),
    )
    assert reply.truncated and reply.mask_sample
    assert reply.info[RESULT_KEY]["recorded_context"]["termination"] == "fixture_gap"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["input", "rubric", "extra", "model", "incomplete"])
async def test_step_identity_faults_never_grade_unbound_evidence(monkeypatch, fault):
    resource, calls = server(monkeypatch)
    await reset(resource)
    row = copy.deepcopy(TASK)
    answer = response(probe())
    if fault == "input":
        row["responses_create_params"]["input"] = []
    elif fault == "rubric":
        row["verifier_metadata"]["partner_if"]["rubrics"][0]["constraint"] += " changed"
    elif fault == "extra":
        row["unexpected"] = "changed"
    elif fault == "model":
        answer["model"] = "other"
    elif fault == "incomplete":
        answer["status"] = "incomplete"
    if fault in {"input", "rubric", "extra"}:
        with pytest.raises(HTTPException, match="changed"):
            await step(resource, answer, row=row)
    else:
        final = await step(resource, answer)
        assert final.mask_sample and final.truncated
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True])
async def test_decisive_last_parent_turn_has_priority_over_the_cap(monkeypatch, passed):
    resource, calls = server(monkeypatch, passed=passed, max_parent_turns=1)
    await reset(resource)
    reply = await step(resource, response(probe()))
    result = reply.info[RESULT_KEY]
    assert len(calls) == 1 and reply.terminated and not reply.truncated and not reply.mask_sample
    assert result["evaluation_completed"] is True and result["reward"] == float(passed)
    assert result["recorded_context"]["termination"] == ("ordering_passed" if passed else "ordering_failed")


@pytest.mark.asyncio
async def test_undecided_cap_masks_and_preserves_the_original_judge_receipt(monkeypatch):
    resource, calls = server(monkeypatch, max_parent_turns=1)
    await reset(resource)
    reply = await step(resource, response(read()))
    result = reply.info[RESULT_KEY]
    assert len(calls) == 1 and reply.mask_sample and reply.truncated and not reply.terminated
    assert result["reward"] == 0 and result["evaluation_completed"] is False
    assert result["recorded_context"]["prefix_history"][0]["judge"]["reward"] == 1
    assert result["recorded_context"]["prefix_history"][0]["judge"]["response"]["status"] == "completed"
    assert result["response"]["status"] == "incomplete"
    assert result["response"]["output"][-1]["type"] == "function_call_output"


@pytest.mark.asyncio
async def test_judge_outage_returns_a_cached_masked_terminal_result(monkeypatch):
    resource, _ = server(monkeypatch)
    await reset(resource)
    attempts = []

    async def unavailable(*args, **kwargs):
        attempts.append(True)
        raise JudgeError("scripted judge unavailable")

    monkeypatch.setattr(app, "grade", unavailable)
    body = response(probe())
    result = await step(resource, body)
    assert len(attempts) == 1 and result.mask_sample
    assert result.info[RESULT_KEY]["failure_kind"] == "judge_failed"
    assert await step(resource, body) == result and len(attempts) == 1


@pytest.mark.asyncio
async def test_cancelled_judgment_is_never_dispatched_again(monkeypatch):
    resource, _ = server(monkeypatch)
    await reset(resource)
    entered = asyncio.Event()
    attempts = []

    async def blocked(*args, **kwargs):
        attempts.append(True)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "grade", blocked)
    body = response(probe())
    running = asyncio.create_task(step(resource, body))
    await asyncio.wait_for(entered.wait(), 2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    reply = await step(resource, body)
    assert len(attempts) == 1 and reply.mask_sample and reply.truncated
    assert reply.info[RESULT_KEY]["recorded_context"]["termination"] == "review_uncertain"


@pytest.mark.asyncio
async def test_optional_close_cancels_an_active_judge_and_is_idempotent(monkeypatch):
    resource, _ = server(monkeypatch)
    ended = []
    monkeypatch.setattr(
        type(resource),
        "checkpoint_session_ended",
        lambda self, incoming: ended.append(incoming.session[SESSION_ID_KEY]),
    )
    await reset(resource)
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "grade", blocked)
    running = asyncio.create_task(step(resource, response(probe())))
    await asyncio.wait_for(entered.wait(), 2)
    await resource._close_endpoint(request())
    with pytest.raises(asyncio.CancelledError):
        await running
    assert resource._episodes["episode"].simulator is None
    assert await resource._close_endpoint(request()) == {"closed": True}
    assert ended == ["episode", "episode"]


@pytest.mark.asyncio
async def test_checkpoint_retirement_releases_only_the_retired_episode(monkeypatch):
    resource, _ = server(monkeypatch)
    await reset(resource)
    await resource._reset_endpoint(EnvResetRequest.model_validate(TASK), request("other"))
    retired = resource._episodes["episode"]
    other = resource._episodes["other"]

    await resource.retire_session_state("episode")
    assert retired.simulator is None and retired.closed_at is not None
    assert other.simulator is not None and other.closed_at is None

    later = retired.closed_at + resource.config.session_close_retry_window_seconds + 1
    monkeypatch.setattr("resources_servers.partner_if.recorded_context_app.time", lambda: later)
    await resource.retire_session_state("episode")
    await resource.retire_session_state("unknown")
    resource._prune()
    assert set(resource._episodes) == set(resource._locks) == {"other"}


@pytest.mark.asyncio
async def test_prefix_observations_are_preserved_without_false_programmatic_grading(monkeypatch):
    resource, calls = server(monkeypatch, observed=True)
    await reset(resource)
    with rollout_context("recorded-context"):
        pending = await step(resource, response(read(), identity="first"), identity="one")
        assert not pending.terminated and not pending.truncated
        final = await step(resource, response(probe(), identity="second"), identity="two")
    result = final.info[RESULT_KEY]
    assert len(calls) == 2 and len(result["ng_verifier_invocations"]) == 2
    assert all(invocation.get("grading_record") is None for invocation in result["ng_verifier_invocations"])
    assert all(
        invocation["grading_operations"][0]["kind"] == "llm_judge" for invocation in result["ng_verifier_invocations"]
    )
    assert result["response"]["usage"]["total_tokens"] == 240
    assert result["responses_create_params"] == PARAMS
    observed = result["ng_agent_observations"]["records"]
    tools = [record for record in observed if record["kind"] == "tool_call"]
    assert len(tools) == 2 and all(record["timing_source"] == "executor" for record in tools)
    assert all(record.get("duration_ms") is None for record in observed if record["kind"] == "agent_invocation")


@pytest.mark.asyncio
async def test_candidate_harness_metadata_must_match_the_server_limits(monkeypatch):
    resource, calls = server(monkeypatch)
    row = copy.deepcopy(TASK)
    row["recorded_context_harness"] = {
        "version": "gymnasium-v1",
        "max_parent_turns": 6,
        "max_policy_calls": 8,
        "max_delegate_turns": 6,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
        "wire_format": "gym-responses-to-chat",
    }
    await reset(resource, row)
    row["recorded_context_harness"]["max_policy_calls"] = 7
    with pytest.raises(HTTPException):
        await resource._reset_endpoint(EnvResetRequest.model_validate(row), request("different"))
    assert not calls
