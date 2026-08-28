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
"""Test terminal attribution: the witnesses join, the anchored builder, and delivery.

The ``/run`` result's ``response`` is the object the verifier scored.
Attribution joins it to exactly one captured call.
The builder then delivers the root-to-terminal chain that earned the reward.
Unattributed rollouts keep the strict single-chain policy bit-for-bit.
"""

import asyncio

from nemo_gym.token_id_capture import TokenCaptureStore, TokenEntry
from nemo_gym.token_id_capture.builder import run_builder
from nemo_gym.token_id_capture.consumer import _assemble
from nemo_gym.token_id_capture.delivery import (
    MASK_SAMPLE_KEY,
    TERMINAL_CALL_KEY,
    TOKEN_CAPTURE_KEY,
    finalize_rollout_token_capture,
)
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


# --- the builder under an attributed terminal ---------------------------------


def _aux_entries() -> list[TokenEntry]:
    """A two-call main chain plus an earlier-completing auxiliary call."""
    return _chain_entries() + [
        # A title-generator call: unrelated prompt, finished first.
        _entry("aux", [90, 91], [92], text="A Title", response_id="resp_aux", created_at=1.0),
    ]


def test_unattributed_aux_call_masks_and_picks_the_wrong_root():
    out = run_builder(_aux_entries())
    assert out.notes.roots == 2 and out.notes.chains == 2
    # Legacy selection picks the earliest root: the auxiliary call.
    main = [c for c in out.chains if c.chain_id == "main"][0]
    assert main.links[0].entry.model_call_id == "aux"


def test_attributed_terminal_delivers_the_verified_chain_and_excludes_aux():
    out = run_builder(_aux_entries(), terminal_call_id="call2")
    assert out.notes.terminal_chain == "delivered"
    main = [c for c in out.chains if c.chain_id == "main"][0]
    assert [link.entry.model_call_id for link in main.links] == ["call1", "call2"]


def test_terminal_truncates_calls_served_after_the_kept_response():
    entries = _chain_entries() + [
        _entry("call3", [1, 2, 3, 4, 5, 6, 7, 8, 9], [10], text="post-terminal", created_at=4.0),
    ]
    out = run_builder(entries, terminal_call_id="call2")
    assert out.notes.terminal_chain == "delivered"
    main = [c for c in out.chains if c.chain_id == "main"][0]
    assert [link.entry.model_call_id for link in main.links] == ["call1", "call2"]


def test_terminal_resolves_a_final_retry_group():
    entries = [
        _entry("call_a", [1, 2], [3, 4], text="answer A", response_id="resp_a"),
        _entry("call_b", [1, 2], [5, 6], text="answer B", response_id="resp_b"),
    ]
    out = run_builder(entries, terminal_call_id="call_b")
    assert out.notes.terminal_chain == "delivered"
    assert out.notes.unresolved_retries == []
    main = [c for c in out.chains if c.chain_id == "main"][0]
    assert [link.entry.model_call_id for link in main.links] == ["call_b"]
    # Without attribution the same shape is unresolved.
    legacy = run_builder(entries)
    assert set(legacy.notes.unresolved_retries) == {"call_a", "call_b"}


def test_terminal_not_captured_is_reported():
    out = run_builder(_chain_entries(), terminal_call_id="ghost")
    assert out.notes.terminal_chain == "not_captured"


# --- consumer mask semantics ---------------------------------------------------


def test_assemble_attributed_aux_rollout_is_not_masked():
    entries = _aux_entries()
    response = _response(
        [_assistant_item("step one"), _assistant_item("final answer")],
        response_id="resp_2",
    )
    built = _assemble("r", entries, "prefix_merging", "m", verified_response=response)
    assert built[MASK_SAMPLE_KEY] is False
    attribution = built["metrics"]["terminal_attribution"]
    assert attribution["method"] == "response_id" and attribution["chain"] == "delivered"
    # The rebuilt response carries only the verified chain's tokens.
    delivered = [i for i in built["rebuilt_response"]["output"] if i.get("generation_token_ids")]
    assert [i["generation_token_ids"] for i in delivered] == [[3, 4], [7, 8]]


def test_assemble_unattributed_aux_rollout_keeps_the_strict_policy():
    built = _assemble("r", _aux_entries(), "prefix_merging", "m", verified_response=None)
    assert built[MASK_SAMPLE_KEY] is True
    assert built["metrics"]["terminal_attribution"]["method"] == "none"


def test_assemble_attributed_but_undeliverable_chain_masks():
    # The named terminal is not among the buildable entries.
    built = _assemble(
        "r",
        _chain_entries(),
        "prefix_merging",
        "m",
        explicit_terminal_call_id="ghost",
        verified_response=None,
    )
    # "ghost" is not captured: no witness lands, so this is unattributed and
    # falls back to the strict policy (single clean chain -> deliverable).
    assert built["metrics"]["terminal_attribution"]["method"] == "none"
    assert built[MASK_SAMPLE_KEY] is False


def test_assemble_broken_terminal_chain_masks():
    # Attribution succeeds but the terminal's ancestry is quarantined.
    e1a = _entry("dup_a", [1, 2], [3, 4], text="same", created_at=1.0)
    e1b = _entry("dup_b", [1, 2], [3, 4], text="same", created_at=1.5)
    child = _entry("child", [1, 2, 3, 4, 9], [10], text="final", response_id="resp_child", created_at=2.0)
    built = _assemble(
        "r",
        [e1a, e1b, child],
        "prefix_merging",
        "m",
        explicit_terminal_call_id="child",
    )
    assert built["metrics"]["terminal_attribution"]["chain"] == "broken"
    assert built[MASK_SAMPLE_KEY] is True


# --- delivery end to end -------------------------------------------------------


def _delivery_case(tmp_path, result: dict) -> dict:
    store = TokenCaptureStore(tmp_path)

    async def go() -> dict:
        for entry in _aux_entries():
            await store.put(entry.model_copy(update={"rollout_id": "t0-r0"}))
        return await finalize_rollout_token_capture(result, store)

    return asyncio.run(go())


def test_finalize_attributes_from_the_result_response(tmp_path):
    result = {
        "_ng_rollout_id": "t0-r0",
        "response": _response(
            [_assistant_item("step one"), _assistant_item("final answer")],
            response_id="resp_2",
        ),
        "reward": 1.0,
    }
    built = _delivery_case(tmp_path, result)
    assert built[MASK_SAMPLE_KEY] is False
    assert result.get(MASK_SAMPLE_KEY) is not True
    attribution = result[TOKEN_CAPTURE_KEY]["terminal_attribution"]
    assert attribution["method"] == "response_id" and attribution["chain"] == "delivered"
    delivered = [i for i in result["response"]["output"] if i.get("generation_token_ids")]
    assert [i["generation_token_ids"] for i in delivered] == [[3, 4], [7, 8]]


def test_finalize_honors_an_explicit_terminal_key(tmp_path):
    result = {
        "_ng_rollout_id": "t0-r0",
        "response": {"id": "", "model": "m", "object": "response", "output": []},
        TERMINAL_CALL_KEY: "call2",
        "reward": 1.0,
    }
    built = _delivery_case(tmp_path, result)
    assert built[MASK_SAMPLE_KEY] is False
    attribution = result[TOKEN_CAPTURE_KEY]["terminal_attribution"]
    assert attribution["method"] == "explicit" and attribution["chain"] == "delivered"


def test_finalize_without_witnesses_keeps_the_strict_policy(tmp_path):
    result = {
        "_ng_rollout_id": "t0-r0",
        "response": {"id": "", "model": "m", "object": "response", "output": []},
        "reward": 1.0,
    }
    built = _delivery_case(tmp_path, result)
    # Two roots (main chain + aux) and no attribution: masked, as before.
    assert built[MASK_SAMPLE_KEY] is True
    assert result[MASK_SAMPLE_KEY] is True
