# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segment rewards mapped to the captured model calls they target."""

from __future__ import annotations

import pytest

from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.models.token_capture.completion import output_item_fingerprint
from nemo_gym.models.token_capture.staging.digest import EMPTY_EXTRAS_DIGEST
from nemo_gym.models.token_capture.staging.records import CallRecord, OutputItemEvidence
from nemo_gym.rollouts.segment_rewards import resolve_segment_rewards
from nemo_gym.trajectory.trajectory_record import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
)


DIGEST = "b" * 64
POLICY = ModelServerRef(type="responses_api_models", name="policy_model")


def _message(text: str) -> dict:
    # Harness transcripts give messages their own ids, so only the content matches the served item.
    return {
        "type": "message",
        "id": f"msg-local-{text}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _tool_call(call_id: str, command: str) -> dict:
    return {
        "type": "function_call",
        "id": call_id,
        "call_id": call_id,
        "name": "Bash",
        "arguments": f'{{"command": "{command}"}}',
        "status": "completed",
    }


def _record(model_call_id: str, *served_items: dict) -> CallRecord:
    return CallRecord(
        model_call_id=model_call_id,
        prev_len=0,
        delta_len=10,
        cum_len=10,
        weight_version=1,
        digest=DIGEST,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        staging_key=f"r1/{model_call_id}",
        mode="text",
        chain_hash=DIGEST,
        cumulative_hash=DIGEST,
        response_id=f"msg_{model_call_id}",
        # Served output keeps item ids only on tool calls.
        output_items=[
            OutputItemEvidence(
                id=item["id"] if item["type"] == "function_call" else None,
                fingerprint=output_item_fingerprint(item),
            )
            for item in served_items
        ],
    )


# One agent whose context was compacted between c2 and c3, so c1-c2 and c3 train as separate segments.
PLAN = _message("I will look at the logs first.")
LS = _tool_call("toolu_ls", "ls")
LAUNCH = _tool_call("toolu_agent", "spawn")
ANSWER = _message("The answer is 42.")
RECORDS = [
    _record("c1", PLAN, LS),
    _record("c2", LAUNCH),
    _record("c3", ANSWER),
    _record("a1"),
    _record("a2"),
]
OUTPUT = [PLAN, LS, {"type": "function_call_output", "call_id": "toolu_ls", "output": "app.log"}, LAUNCH, ANSWER]


def _invocation(invocation_id: str, calls: list[str], *, parent: str | None = None, launched_by: str | None = None):
    return AgentInvocation(
        invocation_id=invocation_id,
        parent_invocation_id=parent,
        spawned_by_tool_call_id=launched_by,
        model_calls=[ModelCallRef(model_ref=POLICY, response_id=f"msg_{call_id}") for call_id in calls],
    )


MAIN = _invocation("main", ["c1", "c2", "c3"])
SUB = _invocation("sub", ["a1", "a2"], parent="main", launched_by="toolu_agent")


def _result(*segment_rewards: dict, observations: tuple = (MAIN, SUB)) -> dict:
    return {
        "reward": 1.0,
        "response": {"id": "msg_c3", "output": OUTPUT},
        "segment_rewards": list(segment_rewards),
        "ng_agent_observations": AgentObservationBundle(source="test", records=list(observations)).model_dump(
            mode="json"
        ),
    }


def test_without_segment_rewards_no_call_is_targeted():
    assert resolve_segment_rewards(RECORDS, {"reward": 1.0}) == {}


def test_a_tool_call_targets_the_call_that_served_it():
    rewards = resolve_segment_rewards(RECORDS, _result({"output_index": 1, "reward": 0.2}))

    assert rewards == {"c1": 0.2}


def test_a_message_targets_the_call_that_wrote_it_by_content():
    rewards = resolve_segment_rewards(RECORDS, _result({"output_index": 4, "reward": 0.9}))

    assert rewards == {"c3": 0.9}


def test_one_agent_and_its_subagent_get_rewards_in_one_response():
    nested = _invocation("sub_child", ["n1"], parent="sub", launched_by="toolu_child")
    records = [*RECORDS, _record("n1")]
    result = _result(
        {"output_index": 0, "reward": 0.2},
        {"launched_by": "toolu_agent", "reward": 0.5},
        observations=(MAIN, SUB, nested),
    )

    # The nested subagent has no entry of its own, so it takes its launcher's.
    assert resolve_segment_rewards(records, result) == {"c1": 0.2, "a1": 0.5, "a2": 0.5, "n1": 0.5}


def test_a_subagents_compaction_call_takes_the_subagents_reward():
    compaction = ContextCompactionObservation(
        invocation_id="sub", model_calls=[ModelCallRef(model_ref=POLICY, response_id="msg_summary")]
    )
    records = [*RECORDS, _record("summary")]
    result = _result({"launched_by": "toolu_agent", "reward": 0.5}, observations=(MAIN, SUB, compaction))

    assert resolve_segment_rewards(records, result)["summary"] == 0.5


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"output_index": 2, "reward": 0.0}, "model-authored"),
        ({"output_index": 9, "reward": 0.0}, "outside the scored response"),
        ({"launched_by": "toolu_missing", "reward": 0.0}, "toolu_missing"),
    ],
    ids=["tool result", "index out of range", "unknown launch"],
)
def test_an_entry_that_targets_nothing_is_rejected(entry, message):
    with pytest.raises(ValueError, match=message):
        resolve_segment_rewards(RECORDS, _result(entry))


def test_an_item_two_calls_could_have_written_is_rejected():
    records = [*RECORDS, _record("c4", ANSWER)]

    with pytest.raises(ValueError, match="matches 2 captured calls"):
        resolve_segment_rewards(records, _result({"output_index": 4, "reward": 0.9}))


def test_two_rewards_for_one_call_are_rejected():
    result = _result({"output_index": 0, "reward": 0.2}, {"output_index": 1, "reward": 0.3})

    with pytest.raises(ValueError, match="c1"):
        resolve_segment_rewards(RECORDS, result)


def test_a_launch_without_agent_observations_is_rejected():
    result = _result({"launched_by": "toolu_agent", "reward": 0.5})
    del result["ng_agent_observations"]

    with pytest.raises(ValueError, match="ng_agent_observations"):
        resolve_segment_rewards(RECORDS, result)


def test_a_launch_whose_subagent_reported_no_calls_is_logged(caplog):
    silent = _invocation("sub", [], parent="main", launched_by="toolu_agent")

    rewards = resolve_segment_rewards(
        RECORDS, _result({"launched_by": "toolu_agent", "reward": 0.5}, observations=(MAIN, silent))
    )

    assert rewards == {}
    assert "toolu_agent" in caplog.text
