# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test terminal attribution: the witnesses join.

The ``/run`` result's ``response`` is the object the verifier scored.
Attribution joins it to exactly one captured call.
The builder then delivers the root-to-terminal chain that earned the reward.
Unattributed rollouts keep the strict single-chain policy bit-for-bit.
"""

from nemo_gym.token_id_capture import TokenEntry
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from nemo_gym.token_id_capture.terminal import resolve_terminal


# --- helpers ------------------------------------------------------------------


def _assistant_item(text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _entry(
    call_id: str,
    prompt: list[int],
    gen: list[int],
    *,
    text: str | None = None,
    response_id: str | None = None,
    created_at: float = 0.0,
    **extra,
) -> TokenEntry:
    output_items = [_assistant_item(text)] if text is not None else []
    return TokenEntry(
        rollout_id="r",
        model_call_id=call_id,
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=[-0.1] * len(gen),
        output_items=output_items,
        token_item_index=0 if output_items else None,
        response_id=response_id,
        created_at=created_at,
        **extra,
    )


def _chain_entries() -> list[TokenEntry]:
    """Two chained calls: call2 continues call1's cumulative sequence."""
    return [
        _entry("call1", [1, 2], [3, 4], text="step one", response_id="resp_1", created_at=2.0),
        _entry("call2", [1, 2, 3, 4, 5, 6], [7, 8], text="final answer", response_id="resp_2", created_at=3.0),
    ]


def _response(items: list[dict], response_id: str = "") -> dict:
    return {"id": response_id, "model": "m", "object": "response", "output": items}


# --- resolve_terminal: the witnesses ------------------------------------------


def test_explicit_witness_attributes():
    entries = _chain_entries()
    att = resolve_terminal(entries, None, explicit_call_id="call2")
    assert att.attributed and att.model_call_id == "call2" and att.method == "explicit"


def test_explicit_terminal_not_captured_abstains():
    att = resolve_terminal(_chain_entries(), None, explicit_call_id="ghost")
    assert not att.attributed
    assert "explicit_terminal_not_captured" in att.reason


def test_response_id_witness_attributes_a_merged_transcript():
    # simple_agent-shaped result: the final call's envelope id with a merged
    # transcript. The trailing-block content reading names the same call.
    entries = _chain_entries()
    response = _response(
        [
            _assistant_item("step one"),
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
            _assistant_item("final answer"),
        ],
        response_id="resp_2",
    )
    att = resolve_terminal(entries, response)
    assert att.attributed and att.model_call_id == "call2" and att.method == "response_id"
    assert "corroborated_by=content_output" in att.reason


def test_trailing_block_attributes_a_merged_transcript_without_ids():
    # The blackbox multi-turn case: a synthesized transcript with no served id.
    # The final block of model-authored items is the terminal call's own output.
    entries = _chain_entries()
    response = _response(
        [
            _assistant_item("step one"),
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
            _assistant_item("final answer"),
        ]
    )
    att = resolve_terminal(entries, response)
    assert att.attributed and att.model_call_id == "call2" and att.method == "content_output"


def test_transcript_ending_in_a_tool_result_skips_the_trailing_reading():
    # A truncated rollout ends with a pending tool result: there is no trailing
    # model-authored block, so the content witness must not match a mid-chain call.
    entries = _chain_entries()
    response = _response(
        [
            _assistant_item("step one"),
            _assistant_item("final answer"),
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
        ]
    )
    att = resolve_terminal(entries, response)
    assert not att.attributed and "no_content_match" in att.reason


def test_repeated_identical_output_abstains_without_an_id():
    # The model produced the same text at two different depths. The trailing
    # reading matches both entries, their token sequences differ, and no id
    # exists to break the tie: abstain and mask rather than guess.
    entries = [
        _entry("call1", [1, 2], [3, 4], text="done", created_at=1.0),
        _entry("call2", [1, 2, 3, 4, 5, 6], [7, 8], text="done", created_at=2.0),
    ]
    response = _response(
        [
            _assistant_item("done"),
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
            _assistant_item("done"),
        ]
    )
    att = resolve_terminal(entries, response)
    assert not att.attributed and "content_ambiguous" in att.reason


def test_content_witness_attributes_a_final_turn_response():
    # A single-turn (or last-response-only) result matches the entry's own output.
    entries = _chain_entries()
    response = _response([_assistant_item("final answer")])
    att = resolve_terminal(entries, response)
    assert att.attributed and att.model_call_id == "call2" and att.method == "content_output"
    assert "response_has_no_id" in att.reason


def test_cumulative_fingerprint_reading_from_a_lineage_aware_writer():
    # A lineage-aware writer stamps continuation_fingerprint (request + own output).
    # The full merged transcript then matches it even though own-output does not.
    transcript = [_assistant_item("step one"), _assistant_item("final answer")]
    target = assistant_fingerprint(transcript)
    entries = [
        _entry("call1", [1, 2], [3, 4], text="step one", created_at=2.0),
        _entry(
            "call2",
            [1, 2, 3, 4, 5, 6],
            [7, 8],
            text="final answer",
            created_at=3.0,
            continuation_fingerprint=target,
        ),
    ]
    att = resolve_terminal(entries, _response(transcript))
    assert att.attributed and att.model_call_id == "call2" and att.method == "content_cumulative"


def test_identical_retries_collapse_to_one_call():
    # Two servings of the same content and tokens are interchangeable for training.
    entries = [
        _entry("call_b", [1, 2], [3, 4], text="same", response_id="resp_b"),
        _entry("call_a", [1, 2], [3, 4], text="same", response_id="resp_a"),
    ]
    att = resolve_terminal(entries, _response([_assistant_item("same")]))
    assert att.attributed and att.model_call_id == "call_a" and att.method == "content_output"


def test_divergent_final_retries_resolve_by_id_and_abstain_on_content():
    # Same prompt, different generations: content cannot say which was kept,
    # but possession of the served id can.
    entries = [
        _entry("call_a", [1, 2], [3, 4], text="answer A", response_id="resp_a"),
        _entry("call_b", [1, 2], [5, 6], text="answer B", response_id="resp_b"),
    ]
    att = resolve_terminal(entries, _response([_assistant_item("answer B")], response_id="resp_b"))
    assert att.attributed and att.model_call_id == "call_b"
    # Both the id and the content witness name call_b; either may lead.
    assert att.method in ("response_id", "content_output")
    assert "corroborated_by=" in att.reason


def test_witness_disagreement_fails_closed():
    entries = _chain_entries()
    # The explicit witness names call1; the response id names call2.
    response = _response([_assistant_item("unrelated")], response_id="resp_2")
    att = resolve_terminal(entries, response, explicit_call_id="call1")
    assert not att.attributed
    assert "witness_disagreement[" in att.reason


def test_a_mutated_echo_matches_nothing():
    # A verifier that rewrites the echoed response breaks the echo contract;
    # the join must fail closed, never land on a near-miss.
    entries = _chain_entries()
    response = _response([_assistant_item("final answer [redacted]")])
    att = resolve_terminal(entries, response)
    assert not att.attributed and "no_content_match" in att.reason


def test_no_response_object_abstains():
    att = resolve_terminal(_chain_entries(), None)
    assert not att.attributed and "no_response_object" in att.reason


def test_duplicated_response_id_abstains_but_content_can_still_attribute():
    # Backend id reuse across different generations is a defect; the id witness
    # abstains and leaves the trail, while content still attributes.
    entries = [
        _entry("call_a", [1, 2], [3, 4], text="answer A", response_id="resp_dup"),
        _entry("call_b", [1, 2], [5, 6], text="answer B", response_id="resp_dup"),
    ]
    att = resolve_terminal(entries, _response([_assistant_item("answer B")], response_id="resp_dup"))
    assert att.attributed and att.model_call_id == "call_b" and att.method == "content_output"
    assert "response_id_ambiguous" in att.reason
