# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real generic environment/replay/verifier boundaries with declared synthetic verdicts."""

import copy
import json
from pathlib import Path

import httpx
import pytest

from benchmarks.partner_if.tests.generic_environment import GenericReplayPair
from benchmarks.partner_if.tests.test_replay import TASK, call, draft, probe, response
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnRequest
from responses_api_agents.ngc_recorded_context.replay import actor_identity
from responses_api_agents.partner_if_dispatcher.app import PartnerIFDispatcher, PartnerIFDispatcherConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pass", "fail", "same_batch", "safe_cutoff", "provider_fault"])
@pytest.mark.parametrize("task_name", ["pif_029", "synthetic-ordering-example"])
@pytest.mark.parametrize("positional_identity", [False, True])
async def test_generic_environment_closes_before_verification_and_preserves_decisive_prefix(
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
        max_activations=1 if mode == "safe_cutoff" else 6,
        policy_responses=queue,
        fixture_passed=mode not in {"fail", "same_batch"},
    )
    async with pair.running():
        assert type(pair.environment) is SingleAgentTurnEnvironmentServer
        reply, request = await pair.run(task_name=task_name, positional_identity=positional_identity)
    assert reply["failure"] is None, reply
    direct = [item for item in pair.trace if item["server"] == "environment"]
    assert len(direct) == 1 and direct[0]["path"] == "/run"
    assert direct[0]["json"] == request.model_dump(mode="json", exclude_unset=True)
    row = reply["result"]
    assert reply["task_id"] == request.task.task_id.model_dump(mode="json")
    assert row["verifier_metadata"]["partner_if"]["task_id"] == "pif_029"
    assert row["mask_sample"] == (mode in {"safe_cutoff", "provider_fault"})
    assert row["evaluation_completed"] == (mode in {"pass", "fail", "same_batch"})
    closes = [i for i, item in enumerate(pair.trace) if item["server"] == "agent" and item["path"].endswith("/close")]
    verifies = [
        i for i, item in enumerate(pair.trace) if item["server"] == "partner_if" and item["path"].endswith("/verify")
    ]
    assert len(closes) == len(verifies)
    assert all(closed < verified for closed, verified in zip(closes, verifies))
    assert pair.trace[-1]["path"].endswith("/close_session")
    assert len(pair.policy_requests) == (2 if mode == "pass" else 1)
    if mode == "pass":
        assert row["reward"] == 1
        assert row["response"]["usage"]["total_tokens"] == 240
        assert len(row["ng_trajectory"]["turns"]) == 2
        assert row["ng_trajectory"]["task_id"] == request.task.task_id.task_id
        activations = [
            item["json"] for item in pair.trace if item["server"] == "agent" and item["path"].endswith("/v1/responses")
        ]
        assert len(activations) == 2
        metadata = activations[1]["metadata"]
        assert metadata["ngc_replay_parent_turn_limit"] == "2"
        saved = Path(metadata["ngc_replay_saved_prefix"])
        receipt = json.loads((saved / "native-step.json").read_text())
        assert receipt["new_policy_calls"] == 1
        fresh_steps = list((tmp_path / "captures").rglob("native-step.json"))
        assert sorted(json.loads(path.read_text())["new_policy_calls"] for path in fresh_steps) == [1, 1]
        assert sorted(json.loads(path.read_text())["reused_policy_calls"] for path in fresh_steps) == [0, 1]
    if mode in {"fail", "same_batch"}:
        assert row["reward"] == 0
    for outbound in pair.policy_requests:
        assert outbound["temperature"] == 0.7 and outbound["top_p"] == 0.95
        assert outbound["max_tokens"] == 8192
        assert not any(key.startswith("ngc_replay_") for key in outbound.get("metadata", {}))


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True])
async def test_first_decisive_prefix_prevents_a_later_fixture_fault(tmp_path, passed):
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
    assert len(pair.policy_requests) == 1 and len(queue) == 1


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
    assert reply["result"].get("next_agent_request") is None
    assert len(pair.policy_requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True])
async def test_dispatcher_runs_the_direct_environment_and_preserves_collection_identity(tmp_path, passed):
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
    assert len(pair.policy_requests) == 1
    direct = [item for item in pair.trace if item["server"] == "environment"]
    assert len(direct) == 1 and direct[0]["path"] == "/run"
    episode = direct[0]["json"]
    assert episode["episode_id"] == {"rollout_id": "27-3", "attempt": 2}
    assert episode["task"]["task_id"] == {"taskset": "agent", "task_id": "27"}
    typed_request = SingleAgentTurnRequest.model_validate(episode)
    task_input = typed_request.task.task_input
    assert task_input.task_data == {key: value for key, value in TASK.items() if key != "responses_create_params"}
    assert actor_identity(task_input.responses_create_params.model_dump(mode="json", exclude_unset=True)) == (
        actor_identity(TASK["responses_create_params"])
    )
    for outbound in pair.policy_requests:
        assert outbound["temperature"] == 0.7 and outbound["top_p"] == 0.95
        assert outbound["max_tokens"] == 8192
