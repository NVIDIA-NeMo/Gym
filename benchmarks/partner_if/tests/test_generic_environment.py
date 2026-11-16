# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stock environment boundaries with real replay captures and declared synthetic grades."""

import asyncio
import copy
import json

import httpx
import pytest

from benchmarks.partner_if.tests.generic_environment import GenericReplayPair
from benchmarks.partner_if.tests.test_replay import TASK, call, draft, probe, response
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from nemo_gym.server_runtime import SingleAgentTurnRequest
from responses_api_agents.ngc_recorded_context.replay import actor_identity
from responses_api_agents.partner_if_dispatcher.app import PartnerIFDispatcher, PartnerIFDispatcherConfig


def calls(pair: GenericReplayPair, service: str, suffix: str) -> list[dict]:
    return [item for item in pair.trace if item["server"] == service and item["path"].endswith(suffix)]


def assert_stock_outer_lifecycle(pair: GenericReplayPair) -> tuple[dict, dict, list[dict]]:
    seeds = calls(pair, "agent", "/v1/agent_sessions")
    activations = calls(pair, "agent", "/v1/responses")
    closes = calls(pair, "agent", "/v1/agent_sessions/close")
    verifies = calls(pair, "partner_if", "/verify")
    assert len(seeds) == len(activations) == len(closes) == len(verifies) == 1
    assert pair.trace.index(activations[0]) < pair.trace.index(closes[0]) < pair.trace.index(verifies[0])
    reviews = calls(pair, "partner_if", "/replay/prefix-review")
    assert reviews and all(pair.trace.index(item) < pair.trace.index(closes[0]) for item in reviews)
    assert all(item["settled_capture_reviewed_before_close"] for item in reviews)
    assert pair.trace[-1]["path"].endswith("/close_session")
    assert "verifier_continuation" not in pair.environment.config.model_dump()
    return seeds[0], activations[0], reviews


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pass", "fail", "same_batch", "safe_cutoff", "provider_fault"])
@pytest.mark.parametrize("task_name", ["pif_029", "synthetic-ordering-example"])
@pytest.mark.parametrize("positional_identity", [False, True])
async def test_stock_environment_uses_one_activation_and_preserves_decisive_prefix(
    tmp_path, mode, task_name, positional_identity
):
    first = response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}))
    queue = (
        [first, response(probe())]
        if mode == "pass"
        else [response(draft())]
        if mode == "fail"
        else [response(probe(), draft())]
        if mode == "same_batch"
        else [TimeoutError("scripted provider fault")]
        if mode == "provider_fault"
        else [first]
    )
    pair = GenericReplayPair(
        tmp_path,
        max_parent_turns=1 if mode == "safe_cutoff" else 6,
        policy_responses=queue,
        fixture_passed=mode not in {"fail", "same_batch"},
    )
    async with pair.running():
        assert type(pair.environment) is SingleAgentTurnEnvironmentServer
        reply, request = await pair.run(task_name=task_name, positional_identity=positional_identity)
    assert reply["failure"] is None, reply
    direct = calls(pair, "environment", "/run")
    assert len(direct) == 1 and direct[0]["json"] == request.model_dump(mode="json", exclude_unset=True)
    row = reply["result"]
    assert reply["task_id"] == request.task.task_id.model_dump(mode="json")
    assert row["verifier_metadata"]["partner_if"]["task_id"] == "pif_029"
    assert row["mask_sample"] == (mode in {"safe_cutoff", "provider_fault"})
    assert row["evaluation_completed"] == (mode in {"pass", "fail", "same_batch"})
    assert "next_agent_request" not in row
    seed, activation, reviews = assert_stock_outer_lifecycle(pair)
    assert actor_identity(activation["json"]) == actor_identity(TASK["responses_create_params"])
    assert seed["json"]["episode_id"] == request.episode_id.model_dump(mode="json")
    assert seed["json"]["task_id"] == request.task.task_id.model_dump(mode="json")
    assert seed["json"]["agent_context"]["values"]["partner_if_replay"] == {
        "task_data": request.task.task_input.task_data
    }
    assert all(item["json"]["responses_create_params"] == activation["json"] for item in reviews)
    assert all(item["json"]["task_data"] == request.task.task_input.task_data for item in reviews)
    assert len(pair.policy_requests) == (2 if mode == "pass" else 1)
    assert len(pair.grade_calls) == (2 if mode == "pass" else 0 if mode == "provider_fault" else 1)
    assert reviews[-1]["reply"]["response"] == row["response"]
    if mode == "pass":
        assert row["reward"] == 1
        assert row["response"]["usage"]["total_tokens"] == 240
        trajectory = row["ng_trajectory"]
        assert len(trajectory["turns"]) == 2 and len(trajectory["tool_calls"]) == 2
        assert trajectory["task_id"] == request.task.task_id.task_id
        assert trajectory["rollout_id"] == request.episode_id.capture_key
        outer = seed["json"]["agent_session_id"]
        assert len([item for item in trajectory["invocations"] if item["invocation_id"] == outer]) == 1
        prefixes = [item for item in trajectory["invocations"] if item.get("parent_invocation_id") == outer]
        assert len(prefixes) == 2
        assert len({item["invocation_id"] for item in prefixes}) == 2
        fresh_steps = list((tmp_path / "captures").rglob("native-step.json"))
        receipts = [json.loads(path.read_text()) for path in fresh_steps]
        assert sorted(item["new_policy_calls"] for item in receipts) == [1, 1]
        assert sorted(item["reused_policy_calls"] for item in receipts) == [0, 1]
    if mode in {"fail", "same_batch"}:
        assert row["reward"] == 0
    for outbound in pair.policy_requests:
        assert outbound["temperature"] == 0.7 and outbound["top_p"] == 0.95
        assert outbound["max_tokens"] == 8192
        assert not any(key.startswith("ngc_replay_") for key in (outbound.get("metadata") or {}))


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True])
async def test_first_decisive_prefix_prevents_another_policy_call_or_judge(tmp_path, passed):
    first = (
        response(probe())
        if passed
        else response(
            call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"}),
            content="Publish another artifact now.",
        )
    )
    queue = [first, TimeoutError("No post-decision provider request is allowed")]
    pair = GenericReplayPair(tmp_path, policy_responses=queue, fixture_passed=passed)
    async with pair.running():
        reply, _ = await pair.run()
    assert reply["failure"] is None
    assert reply["result"]["reward"] == float(passed)
    assert reply["result"]["evaluation_completed"] is True
    assert reply["result"]["mask_sample"] is False
    assert len(pair.policy_requests) == len(pair.grade_calls) == 1 and len(queue) == 1
    assert_stock_outer_lifecycle(pair)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["safe-final", "gap-in-probe-batch"])
async def test_a_nonviolating_reply_or_an_incomplete_probe_batch_cannot_earn_a_pass(tmp_path, case):
    first = (
        response(content="I am investigating.")
        if case == "safe-final"
        else response(probe(), call("Read", {"file_path": "/missing"}, "missing"))
    )
    pair = GenericReplayPair(tmp_path, policy_responses=[first], fixture_passed=True)
    async with pair.running():
        reply, _ = await pair.run()
    assert reply["failure"] is None
    assert reply["result"]["mask_sample"] is True
    assert reply["result"]["evaluation_completed"] is False
    assert "next_agent_request" not in reply["result"]
    assert len(pair.policy_requests) == 1
    assert_stock_outer_lifecycle(pair)


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True])
async def test_dispatcher_runs_the_stock_environment_and_preserves_collection_identity(tmp_path, passed):
    pair = GenericReplayPair(
        tmp_path,
        policy_responses=[response(probe() if passed else draft())],
        fixture_passed=passed,
    )
    row = copy.deepcopy(TASK) | {
        "agent_ref": {"name": "dispatcher"},
        "task_source": "dispatcher",
        "_ng_task_index": 27,
        "_ng_rollout_index": 3,
        "_ng_attempt_index": 2,
    }
    async with pair.running():
        dispatcher = PartnerIFDispatcher(
            config=PartnerIFDispatcherConfig(
                name="dispatcher",
                host="127.0.0.1",
                port=1,
                entrypoint="app.py",
                task_routes={"pif_029": {"type": "environment_servers", "name": "environment"}},
                resources_server={"type": "resources_servers", "name": "partner_if"},
            ),
            server_client=pair.client,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=dispatcher.setup_webserver()), base_url="http://dispatcher"
        ) as client:
            result = await client.post("/run", json=row)
    assert result.status_code == 200, result.text
    flat = result.json()
    assert flat["agent_ref"] == {"name": "agent"}
    assert flat["reward"] == float(passed)
    assert flat["mask_sample"] is False and flat["evaluation_completed"] is True
    assert flat["ng_trajectory"]["task_id"] == "27"
    assert flat["ng_trajectory"]["rollout_id"] == "27-3-a2"
    assert len(pair.policy_requests) == len(pair.grade_calls) == 1
    direct = calls(pair, "environment", "/run")
    assert len(direct) == 1
    episode = direct[0]["json"]
    assert episode["episode_id"] == {"rollout_id": "27-3", "attempt": 2}
    assert episode["task"]["task_id"] == {"taskset": "agent", "task_id": "27"}
    task_input = SingleAgentTurnRequest.model_validate(episode).task.task_input
    assert task_input.task_data == {key: value for key, value in TASK.items() if key != "responses_create_params"}
    assert task_input.agent_seed_context.values["partner_if_replay"] == {"task_data": task_input.task_data}
    assert actor_identity(task_input.responses_create_params.model_dump(mode="json", exclude_unset=True)) == (
        actor_identity(TASK["responses_create_params"])
    )
    assert_stock_outer_lifecycle(pair)


@pytest.mark.asyncio
async def test_identical_prefix_review_retry_reuses_its_grade_and_terminal_receipt(tmp_path):
    pair = GenericReplayPair(tmp_path, policy_responses=[response(probe())], repeat_prefix_review=True)
    async with pair.running():
        reply, _ = await pair.run()
    assert reply["failure"] is None and reply["result"]["reward"] == 1
    reviews = calls(pair, "partner_if", "/replay/prefix-review")
    assert len(reviews) == 2 and reviews[1]["explicit_identical_retry"]
    assert reviews[0]["reply"] == reviews[1]["reply"]
    assert len(pair.grade_calls) == len(pair.policy_requests) == 1
    assert len(calls(pair, "partner_if", "/verify")) == 1


@pytest.mark.asyncio
async def test_lost_outer_activation_reply_reuses_one_activation_without_duplicate_policy_or_grading(tmp_path):
    pair = GenericReplayPair(tmp_path, policy_responses=[response(probe())], lose_activation_reply=True)
    async with pair.running():
        reply, _ = await pair.run()
    assert reply["failure"] is None and reply["result"]["reward"] == 1
    activations = calls(pair, "agent", "/v1/responses")
    assert len(activations) == 2 and activations[0]["reply_dropped_after_execution"]
    assert activations[0]["json"] == activations[1]["json"]
    assert actor_identity(activations[0]["json"]) == actor_identity(TASK["responses_create_params"])
    assert len(pair.policy_requests) == len(pair.grade_calls) == 1
    assert len(calls(pair, "agent", "/v1/agent_sessions")) == 1
    assert len(calls(pair, "agent", "/v1/agent_sessions/close")) == 1
    assert len(calls(pair, "partner_if", "/verify")) == 1


@pytest.mark.asyncio
async def test_lost_prefix_assessment_reply_does_not_repeat_policy_or_claim_an_unreceived_grade(tmp_path):
    pair = GenericReplayPair(tmp_path, policy_responses=[response(probe())], lose_prefix_review_reply=True)
    async with pair.running():
        reply, _ = await pair.run()
    assert reply["failure"] is None
    assert reply["result"]["mask_sample"] is True and reply["result"]["evaluation_completed"] is False
    assert len(pair.policy_requests) == len(pair.grade_calls) == 1
    assert len(calls(pair, "partner_if", "/replay/prefix-review")) == 1
    assert list((tmp_path / "captures").rglob("native-step.json"))
    assert len(calls(pair, "agent", "/v1/agent_sessions/close")) == 1


@pytest.mark.asyncio
async def test_cancellation_closes_the_single_session_and_retains_completed_prefix_evidence(tmp_path):
    pair = GenericReplayPair(
        tmp_path,
        policy_responses=[
            response(call("Read", {"file_path": "/opt/agent/skills/registry-access/SKILL.md"})),
            response(probe()),
        ],
        block_policy_after=1,
    )
    async with pair.running():
        run = asyncio.create_task(pair.environment.run_request(pair.episode_request()))
        try:
            await asyncio.wait_for(pair.policy_blocked.wait(), timeout=20)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run, timeout=20)
        finally:
            pair.release_policy.set()
            if not run.done():
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)
        records = list(pair.agent._session_records.values())
        assert len(records) == 1 and records[0].state is None
        close = records[0].close_response
        assert close.cleanup_confirmed is True
        assert close.agent_observations.records and close.trajectory.turns
    assert len(pair.policy_requests) == 2 and len(pair.grade_calls) == 1
    assert len(calls(pair, "agent", "/v1/agent_sessions/close")) == 1
    assert not calls(pair, "partner_if", "/verify")
    assert list((tmp_path / "captures").rglob("native-step.json"))
