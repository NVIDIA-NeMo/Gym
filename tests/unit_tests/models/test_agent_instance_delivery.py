# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Delivery of each agent instance's rebuilt response into the result row's ``agent_instances`` entries.

Covers how a declared terminal response id is used per agent instance, including a response id that several
agent instances hold, and what the row holds when an agent instance lost every call or when no primary agent
instance was captured.
"""

import pytest

from nemo_gym.models.token_capture.delivery import MASK_SAMPLE_KEY, TOKEN_CAPTURE_KEY, finalize_rollout_token_capture
from nemo_gym.models.token_capture.lineage import stamp_continuation
from nemo_gym.models.token_capture.records import (
    CallIdentity,
    ModelRef,
    ParentResolutionStatus,
    TokenEntry,
    stamp_lineage,
)
from nemo_gym.models.token_capture.store import TokenCaptureStore


ROLLOUT = "r"
POLICY = ModelRef(name="policy")


def _call(model_call_id: str, agent_instance: str | None, prompt: list[int], generation: list[int]) -> TokenEntry:
    """A root call whose served response id is ``resp_<model_call_id>``."""
    entry = TokenEntry(
        capture_key=ROLLOUT,
        model_call_id=model_call_id,
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        generation_log_probs=[-0.1] * len(generation),
        output_items=[{"type": "message", "role": "assistant", "content": model_call_id}],
        response_id=f"resp_{model_call_id}",
        agent_instance=agent_instance,
        model_ref=POLICY,
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    return stamp_continuation(entry, [{"role": "user", "content": model_call_id}])


def _store(tmp_path, *entries: TokenEntry) -> TokenCaptureStore:
    store = TokenCaptureStore(tmp_path)
    for entry in entries:
        store.append(entry)
    return store


def _row(**fields) -> dict:
    return {"_ng_rollout_id": ROLLOUT, "reward": 1.0, "response": {"output": []}, **fields}


def _generation_tokens(response: dict) -> list[list[int]]:
    return [item["generation_token_ids"] for item in response["output"]]


async def test_a_declared_terminal_selects_among_an_agent_instance_s_own_chains(tmp_path):
    # The judge made two unrelated root calls. Alone they are ambiguous; its declared terminal picks one.
    entries = (
        _call("a1", "assistant", [1, 2], [3]),
        _call("j1", "judge", [7, 8], [9]),
        _call("j2", "judge", [5, 6], [4]),
    )
    declared = _row(agent_instances={"judge": {"terminal_response_id": "resp_j2"}})
    undeclared = _row()

    await finalize_rollout_token_capture(declared, _store(tmp_path / "declared", *entries))
    await finalize_rollout_token_capture(undeclared, _store(tmp_path / "undeclared", *entries))

    judge = declared["agent_instances"]["judge"]
    assert judge[MASK_SAMPLE_KEY] is False
    assert _generation_tokens(judge["response"]) == [[4]]
    assert judge[TOKEN_CAPTURE_KEY]["terminal_attribution"]["call_id"] == "j2"
    assert judge["terminal_response_id"] == "resp_j2"
    # Without the declared terminal the same capture cannot choose between the judge's two chains.
    assert undeclared["agent_instances"]["judge"][MASK_SAMPLE_KEY] is True
    # Either way the assistant's sample is its own.
    for row in (declared, undeclared):
        assert row.get(MASK_SAMPLE_KEY) is not True
        assert _generation_tokens(row["response"]) == [[3]]


async def test_the_primary_agent_instance_s_own_entry_attributes_the_top_level_response(tmp_path):
    # No top-level terminal_response_id: the assistant's agent_instances entry names its terminal.
    store = _store(tmp_path, _call("a1", "assistant", [1, 2], [3]), _call("a2", "assistant", [5, 6], [4]))
    row = _row(agent_instances={"assistant": {"terminal_response_id": "resp_a2", "reward": 1.0}})

    await finalize_rollout_token_capture(row, store)

    assert row.get(MASK_SAMPLE_KEY) is not True
    assert _generation_tokens(row["response"]) == [[4]]
    assert _generation_tokens(row["agent_instances"]["assistant"]["response"]) == [[4]]
    assert row["agent_instances"]["assistant"]["reward"] == 1.0


async def test_the_top_level_terminal_outranks_the_primary_agent_instance_s_entry(tmp_path):
    store = _store(tmp_path, _call("a1", "assistant", [1, 2], [3]), _call("a2", "assistant", [5, 6], [4]))
    row = _row(terminal_response_id="resp_a1", agent_instances={"assistant": {"terminal_response_id": "resp_a2"}})

    await finalize_rollout_token_capture(row, store)

    assert row.get(MASK_SAMPLE_KEY) is not True
    assert _generation_tokens(row["response"]) == [[3]]
    assert row[TOKEN_CAPTURE_KEY]["terminal_attribution"]["call_id"] == "a1"


async def test_an_agent_instance_that_lost_every_call_gets_a_masked_entry_without_a_response(tmp_path):
    store = _store(tmp_path, _call("a1", "assistant", [1, 2], [3]))
    # The judge registered a call that never committed a record.
    store._begin_call(ROLLOUT, "j1", CallIdentity(agent_instance="judge", model_ref=POLICY))
    row = _row(
        agent_instances={
            "assistant": {"terminal_response_id": "resp_a1"},
            "judge": {"terminal_response_id": "resp_j1", "reward": 0.25},
        }
    )

    await finalize_rollout_token_capture(row, store)

    judge = row["agent_instances"]["judge"]
    assert "response" not in judge
    assert judge[MASK_SAMPLE_KEY] is True
    assert judge[TOKEN_CAPTURE_KEY]["error"] == "capture contains no token records"
    assert judge[TOKEN_CAPTURE_KEY]["capture_incomplete"] is True
    # What the environment reported for the judge stays beside the failed build.
    assert judge["terminal_response_id"] == "resp_j1" and judge["reward"] == 0.25
    # The judge's loss is its own: the assistant's sample is delivered unmasked.
    assistant = row["agent_instances"]["assistant"]
    assert row.get(MASK_SAMPLE_KEY) is not True and assistant[MASK_SAMPLE_KEY] is False
    assert _generation_tokens(row["response"]) == [[3]]
    assert "capture_incomplete" not in assistant[TOKEN_CAPTURE_KEY]


async def test_a_captured_agent_instance_is_delivered_when_no_primary_agent_instance_was_captured(tmp_path):
    # ``training_agent_instances: [judge]`` captures the judge only: the top-level build is masked, and the
    # judge's own build must still reach the row, which the environment gave no agent_instances map.
    store = _store(tmp_path, _call("j1", "judge", [7, 8], [9]))
    scored_output = [{"type": "message", "role": "assistant", "content": "the answer is 4"}]
    row = {
        "_ng_rollout_id": ROLLOUT,
        "reward": 0.75,
        "response": {"id": "resp_assistant", "output": [dict(item) for item in scored_output]},
    }

    await finalize_rollout_token_capture(row, store)

    assert row[MASK_SAMPLE_KEY] is True and row["response"]["output"] == scored_output
    assert row[TOKEN_CAPTURE_KEY]["error"] == "no primary agent instance captured"
    assert set(row["agent_instances"]) == {"judge"}
    judge = row["agent_instances"]["judge"]
    assert judge[MASK_SAMPLE_KEY] is False
    assert _generation_tokens(judge["response"]) == [[9]]
    assert "error" not in judge[TOKEN_CAPTURE_KEY]


async def test_a_single_agent_capture_adds_no_agent_instances_to_the_row(tmp_path):
    store = _store(tmp_path, _call("c1", None, [1, 2], [3]))
    row = _row()

    await finalize_rollout_token_capture(row, store)

    assert "agent_instances" not in row
    assert row.get(MASK_SAMPLE_KEY) is not True and _generation_tokens(row["response"]) == [[3]]


def _call_on(server: str, model_call_id: str, agent_instance: str | None, response_id: str, generation: list[int]):
    """A root call served by the model server ``server`` under ``response_id``."""
    entry = _call(model_call_id, agent_instance, [1, 2], generation)
    return entry.model_copy(update={"model_ref": ModelRef(name=server), "response_id": response_id})


# A model server numbers its own responses, so calls of different agent instances may carry the same response id.
_ASSISTANT_CALL = ("policy", "a1", "assistant", "resp_1", [3])
_UNNAMED_CALL = ("policy", "c1", None, "resp_1", [3])
_JUDGE_CALL = ("judge-model", "j1", "judge", "resp_1", [9])
_USER_CALL = ("user-model", "u1", "user", "resp_u1", [5])


@pytest.mark.parametrize("reverse_capture_order", [False, True])
@pytest.mark.parametrize(
    "calls, row_fields, conflicts",
    [
        pytest.param(
            (_ASSISTANT_CALL, _JUDGE_CALL),
            {
                "agent_instances": {
                    "assistant": {"terminal_response_id": "resp_1"},
                    "judge": {"terminal_response_id": "resp_1"},
                }
            },
            {},
            id="each-agent-instance-declares-its-own-call",
        ),
        pytest.param(
            (_UNNAMED_CALL, _JUDGE_CALL),
            {"terminal_response_id": "resp_1"},
            {},
            id="the-unnamed-agent-declares-its-own-call-at-the-top-level",
        ),
        pytest.param(
            (_ASSISTANT_CALL, _JUDGE_CALL, _USER_CALL),
            {"agent_instances": {"user": {"terminal_response_id": "resp_1"}}},
            {"user": "assistant"},
            id="a-third-agent-instance-declares-it",
        ),
        pytest.param(
            (_UNNAMED_CALL, _JUDGE_CALL, _USER_CALL),
            {"agent_instances": {"user": {"terminal_response_id": "resp_1"}}},
            {"user": "judge"},
            id="a-third-agent-instance-declares-it-beside-the-unnamed-agent",
        ),
    ],
)
async def test_a_response_id_several_agent_instances_hold_conflicts_only_for_one_that_does_not_hold_it(
    tmp_path, calls, row_fields, conflicts, reverse_capture_order
):
    captured = reversed(calls) if reverse_capture_order else calls
    row = _row(**row_fields)

    await finalize_rollout_token_capture(row, _store(tmp_path, *(_call_on(*call) for call in captured)))

    # The top-level sample is the primary agent instance's, whichever other agent instance holds the same id.
    assert row.get(MASK_SAMPLE_KEY) is not True
    assert "terminal_conflict" not in row[TOKEN_CAPTURE_KEY]
    assert _generation_tokens(row["response"]) == [[3]]
    expected_generation = {"assistant": [[3]], "judge": [[9]], "user": [[5]]}
    for name, entry in row["agent_instances"].items():
        # A conflict names the same holder in either capture order; without one the entry is its own sample.
        assert entry[TOKEN_CAPTURE_KEY].get("terminal_conflict") == conflicts.get(name), name
        assert ("terminal_conflict" in entry[TOKEN_CAPTURE_KEY]) is (name in conflicts), name
        assert entry[MASK_SAMPLE_KEY] is (name in conflicts), name
        assert _generation_tokens(entry["response"]) == expected_generation[name], name
