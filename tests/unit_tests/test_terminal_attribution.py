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
Unattributed rollouts without a declared terminal response ID keep the strict
single-chain policy. An unconfirmed declaration masks the rollout instead.
"""

import asyncio
from pathlib import Path

import pytest

from nemo_gym.token_id_capture import (
    ParentResolutionStatus,
    RefusalRecord,
    TokenCaptureStore,
    TokenEntry,
    stamp_lineage,
)
from nemo_gym.token_id_capture.builder import run_builder
from nemo_gym.token_id_capture.consumer import _assemble
from nemo_gym.token_id_capture.delivery import (
    MASK_SAMPLE_KEY,
    TERMINAL_CALL_KEY,
    TERMINAL_RESPONSE_ID_KEY,
    TOKEN_CAPTURE_KEY,
    capture_build_can_retire,
    finalize_rollout_token_capture,
)
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from nemo_gym.token_id_capture.terminal import resolve_terminal


# --- helpers ------------------------------------------------------------------


def _assistant_item(text: str, item_id: str | None = None) -> dict:
    item = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    if item_id:
        item["id"] = item_id
    return item


def _tool_call_item(call_id: str) -> dict:
    return {
        "type": "function_call",
        "id": call_id,
        "call_id": call_id,
        "name": "exec_command",
        "arguments": '{"cmd": "ls"}',
        "status": "completed",
    }


def _tool_call_entry(
    call_id: str, prompt: list[int], gen: list[int], tool_call_id: str, *, chat: bool = False
) -> TokenEntry:
    """An entry whose recorded output is one tool call: a Responses ``function_call`` item, or with
    ``chat`` the chat form, where the call is nested in the assistant message's ``tool_calls``."""
    if chat:
        item = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd": "ls"}'},
                }
            ],
        }
    else:
        item = _tool_call_item(tool_call_id)
    entry = TokenEntry(
        rollout_id="r",
        model_call_id=call_id,
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=[-0.1] * len(gen),
        output_items=[item],
        token_item_index=0,
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    return entry


def _entry(
    call_id: str,
    prompt: list[int],
    gen: list[int],
    *,
    text: str | None = None,
    response_id: str | None = None,
    created_at: float = 0.0,
    parent_call_id: str | None = None,
    **extra,
) -> TokenEntry:
    output_items = [_assistant_item(text)] if text is not None else []
    entry = TokenEntry(
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
    status = ParentResolutionStatus.RESOLVED if parent_call_id else ParentResolutionStatus.ROOT
    stamp_lineage(entry, parent_call_id, parent_resolution=status)
    return entry


def _chain_entries() -> list[TokenEntry]:
    """Two chained calls: call2 continues call1's cumulative sequence."""
    return [
        _entry("call1", [1, 2], [3, 4], text="step one", response_id="resp_1", created_at=2.0),
        _entry(
            "call2",
            [1, 2, 3, 4, 5, 6],
            [7, 8],
            text="final answer",
            response_id="resp_2",
            created_at=3.0,
            parent_call_id="call1",
        ),
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


def test_item_id_witness_attributes_a_transcript_ending_in_a_tool_result():
    """A session killed while a tool ran ends its transcript on that tool's output. The last
    model-authored item is the tool call before that output, and its served call id names the
    entry. The response id cannot, because the agent server assigned it. The content readings
    cannot either, because they skip a transcript that ends in a tool result."""
    a = _entry("a", [1, 2], [3], text="plan")
    a.output_items[0]["id"] = "msg_a"
    b = _tool_call_entry("b", [1, 2, 3, 4], [5], "chatcmpl-tool-b")
    response = {
        "id": "resp_minted_by_the_agent_server",
        "output": [
            _assistant_item("plan", item_id="msg_a"),
            _tool_call_item("chatcmpl-tool-b"),
            {"type": "function_call_output", "call_id": "chatcmpl-tool-b", "output": "ok"},
        ],
    }
    verdict = resolve_terminal([a, b], response)
    assert (verdict.model_call_id, verdict.method) == ("b", "item_id")
    assert "response_id_no_match" in verdict.reason


def test_item_id_witness_uses_a_served_message_id_and_content_corroborates():
    a = _entry("a", [1, 2], [3], text="plan")
    b = _entry("b", [1, 2, 3, 4], [5], text="done")
    b.output_items[0]["id"] = "msg_b"
    response = {
        "output": [
            _assistant_item("plan"),
            {"type": "function_call_output", "call_id": "x", "output": "ok"},
            _assistant_item("done", item_id="msg_b"),
        ]
    }
    verdict = resolve_terminal([a, b], response)
    assert (verdict.model_call_id, verdict.method) == ("b", "item_id")
    assert "corroborated_by=content_output" in verdict.reason


def test_a_message_the_harness_composed_does_not_attribute_by_item_id():
    """A closing message written by the agent server carries no served id, so the witness
    abstains instead of naming an earlier call."""
    a = _entry("a", [1, 2], [3], text="plan", response_id="resp_a")
    response = {
        "output": [_assistant_item("plan"), _assistant_item("Submitting ./kernel.py", item_id="msg-submit-1234")]
    }
    verdict = resolve_terminal([a], response)
    assert not verdict.attributed
    assert "item_id_no_match" in verdict.reason


def test_item_id_disagreeing_with_the_response_id_fails_closed():
    a = _entry("a", [1, 2], [3], text="plan", response_id="resp_a")
    b = _entry("b", [1, 2, 3, 4], [5], text="done")
    b.output_items[0]["id"] = "msg_b"
    response = {"id": "resp_a", "output": [_assistant_item("plan"), _assistant_item("done", item_id="msg_b")]}
    verdict = resolve_terminal([a, b], response)
    assert not verdict.attributed
    assert "witness_disagreement" in verdict.reason


def test_item_id_witness_reads_tool_calls_nested_in_a_chat_message():
    a = _entry("a", [1, 2], [3], text="plan")
    b = _tool_call_entry("b", [1, 2, 3, 4], [5], "chatcmpl-tool-b", chat=True)
    response = {
        "output": [
            _assistant_item("plan"),
            _tool_call_item("chatcmpl-tool-b"),
            {"type": "function_call_output", "call_id": "chatcmpl-tool-b", "output": "ok"},
        ]
    }
    verdict = resolve_terminal([a, b], response)
    assert (verdict.model_call_id, verdict.method) == ("b", "item_id")


def test_item_id_witness_ignores_the_namespace_prefix_on_a_flattened_tool_name():
    """A namespaced Responses tool reaches the chat backend as ``<namespace>__<name>`` and is
    recorded that way; codex's rollout file keeps the bare name, so a transcript ending on such a
    call must still match the entry by served id and content."""
    a = _entry("a", [1, 2], [3], text="plan")
    b = TokenEntry(
        rollout_id="r",
        model_call_id="b",
        prompt_token_ids=[1, 2, 3, 4],
        generation_token_ids=[5],
        generation_log_probs=[-0.1],
        output_items=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-ns",
                        "type": "function",
                        "function": {"name": "multi_agent_v1__send_input", "arguments": '{"id": "agent-1"}'},
                    }
                ],
            }
        ],
        token_item_index=0,
    )
    stamp_lineage(b, None, parent_resolution=ParentResolutionStatus.ROOT)
    response = {
        "output": [
            _assistant_item("plan"),
            {
                "type": "function_call",
                "id": "chatcmpl-tool-ns",
                "call_id": "chatcmpl-tool-ns",
                "name": "send_input",
                "arguments": '{"id": "agent-1"}',
                "status": "completed",
            },
            {"type": "function_call_output", "call_id": "chatcmpl-tool-ns", "output": "sent"},
        ]
    }
    verdict = resolve_terminal([a, b], response)
    assert (verdict.model_call_id, verdict.method) == ("b", "item_id")


def test_item_id_shared_by_two_different_calls_abstains():
    a = _tool_call_entry("a", [1, 2], [3], "chatcmpl-tool-x")
    b = _tool_call_entry("b", [1, 2, 3, 4], [5], "chatcmpl-tool-x")
    response = {
        "output": [
            _tool_call_item("chatcmpl-tool-x"),
            {"type": "function_call_output", "call_id": "chatcmpl-tool-x", "output": "ok"},
        ]
    }
    verdict = resolve_terminal([a, b], response)
    assert not verdict.attributed
    assert "item_id_ambiguous" in verdict.reason


def test_last_item_without_a_served_id_abstains():
    a = _entry("a", [1, 2], [3], text="plan")
    response = {
        "output": [
            _assistant_item("plan"),
            {"type": "function_call_output", "call_id": "x", "output": "ok"},
            _assistant_item("done"),
        ]
    }
    verdict = resolve_terminal([a], response)
    assert "last_item_has_no_id" in verdict.reason


def test_item_id_with_different_content_does_not_attribute():
    """A reused call id whose recorded arguments differ from the transcript's last tool call
    names nothing: attributing it would train a chain that lacks the real terminal turn."""
    a = _tool_call_entry("a", [1, 2], [3], "call_0")
    last = {**_tool_call_item("call_0"), "arguments": '{"cmd": "rm -rf build"}'}
    response = {
        "output": [
            _tool_call_item("call_0"),
            {"type": "function_call_output", "call_id": "call_0", "output": "ok"},
            last,
            {"type": "function_call_output", "call_id": "call_0", "output": "done"},
        ]
    }
    verdict = resolve_terminal([a], response)
    assert not verdict.attributed
    assert "item_id_content_mismatch" in verdict.reason


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
        _entry(
            "call3",
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10],
            text="post-terminal",
            created_at=4.0,
            parent_call_id="call2",
        ),
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
    child = _entry(
        "child",
        [1, 2, 5, 6, 9],
        [10],
        text="final",
        response_id="resp_child",
        created_at=2.0,
        parent_call_id="dup_a",
    )
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


def _delivery_case(tmp_path, result: dict, *, with_aux: bool = True) -> dict:
    store = TokenCaptureStore(tmp_path)

    async def go() -> dict:
        for entry in _aux_entries() if with_aux else _chain_entries():
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


def _length_cut_chain_entries() -> list[TokenEntry]:
    """Two chained calls whose terminal call stopped at the output-token budget."""
    return [
        _entry("call1", [1, 2], [3, 4], text="step one", response_id="resp_1", created_at=2.0),
        _entry(
            "call2",
            [1, 2, 3, 4, 5, 6],
            [7, 8],
            text="final answer",
            response_id="resp_2",
            created_at=3.0,
            parent_call_id="call1",
            finish_reason="length",
        ),
    ]


def _finalize_length_cut_rollout(tmp_path, harness_status: str | None) -> tuple[dict, dict]:
    """Deliver a length-cut rollout whose harness response reports ``harness_status`` (or none)."""
    store = TokenCaptureStore(tmp_path)
    response = _response([_assistant_item("step one"), _assistant_item("final answer")], response_id="resp_2")
    if harness_status is not None:
        response["status"] = harness_status
    result = {"_ng_rollout_id": "t0-r0", "response": response, "reward": 1.0}

    async def go() -> dict:
        for entry in _length_cut_chain_entries():
            await store.put(entry.model_copy(update={"rollout_id": "t0-r0"}))
        return await finalize_rollout_token_capture(result, store)

    return asyncio.run(go()), result


@pytest.mark.parametrize("harness_status", ["completed", None])
def test_finalize_marks_the_delivered_response_incomplete_when_the_terminal_call_was_cut(tmp_path, harness_status):
    # The agent server's response says "completed" (or says nothing) because the harness ended the
    # rollout on its own terms, but the capture shows the terminal call stopped at the output
    # budget. Delivery carries the builder's verdict onto the result's response, so a consumer
    # reading ``incomplete_details`` sees the truncation that the rebuilt output reflects.
    built, result = _finalize_length_cut_rollout(tmp_path, harness_status)
    assert built[MASK_SAMPLE_KEY] is False
    assert built["rebuilt_response"]["status"] == "incomplete"
    assert result["response"]["status"] == "incomplete"
    assert result["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    assert [i["generation_token_ids"] for i in result["response"]["output"] if i.get("generation_token_ids")] == [
        [3, 4],
        [7, 8],
    ]


def test_finalize_keeps_a_harness_verdict_that_is_already_a_failure(tmp_path):
    # A harness that reported ``failed`` gave a more specific verdict than the capture's truncation;
    # delivery replaces the output but leaves that status and its absent ``incomplete_details``.
    built, result = _finalize_length_cut_rollout(tmp_path, "failed")
    assert built["rebuilt_response"]["status"] == "incomplete"
    assert result["response"]["status"] == "failed"
    assert "incomplete_details" not in result["response"]
    assert [i["generation_token_ids"] for i in result["response"]["output"] if i.get("generation_token_ids")] == [
        [3, 4],
        [7, 8],
    ]


def _completed_chain_entries() -> list[TokenEntry]:
    """Two chained calls whose terminal call completed on its own."""
    return [
        _entry("call1", [1, 2], [3, 4], text="step one", response_id="resp_1", created_at=2.0),
        _entry(
            "call2",
            [1, 2, 3, 4, 5, 6],
            [7, 8],
            text="final answer",
            response_id="resp_2",
            created_at=3.0,
            parent_call_id="call1",
        ),
    ]


@pytest.mark.parametrize(
    ("refused_at", "expected_status"),
    [(3.5, "incomplete"), (2.5, None)],
    ids=["terminal-refusal", "recovered-refusal"],
)
def test_a_refusal_ends_the_rebuilt_response_only_when_it_came_after_the_terminal_call(refused_at, expected_status):
    # A refusal after the last served call means the harness's next request no longer fit the
    # context window: the rollout ended at the model's limit. A refusal the harness recovered
    # from, by compacting and getting a later call served, ended nothing.
    built = _assemble(
        "r",
        _completed_chain_entries(),
        "prefix_merging",
        "m",
        explicit_terminal_call_id="call2",
        refusals=(RefusalRecord(model_call_id="call3", code="context_length_exceeded", created_at=refused_at),),
    )
    assert built[MASK_SAMPLE_KEY] is False
    assert built["rebuilt_response"].get("status") == expected_status
    expected_details = {"reason": "max_output_tokens"} if expected_status else None
    assert built["rebuilt_response"].get("incomplete_details") == expected_details


def test_finalize_reports_a_terminal_refusal_on_the_result(tmp_path):
    # The store records the refusal and resolves the refused call's intent, so the frozen
    # snapshot is complete; the refusal reaches delivery as the builder's verdict and as a
    # metrics entry that distinguishes it from a served ``length`` stop.
    store = TokenCaptureStore(tmp_path)
    response = _response([_assistant_item("step one"), _assistant_item("final answer")], response_id="resp_2")
    result = {"_ng_rollout_id": "t0-r0", "response": response, "reward": 1.0}

    async def go() -> dict:
        for entry in _completed_chain_entries():
            await store.put(entry.model_copy(update={"rollout_id": "t0-r0"}))
        await store.begin_call("t0-r0", "call3")
        await store.refuse_call("t0-r0", model_call_id="call3", code="context_length_exceeded")
        return await finalize_rollout_token_capture(result, store)

    built = asyncio.run(go())
    assert built[MASK_SAMPLE_KEY] is False
    assert result.get(MASK_SAMPLE_KEY) is not True
    assert result["response"]["status"] == "incomplete"
    assert result["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    (refusal,) = result[TOKEN_CAPTURE_KEY]["refusals"]
    assert (refusal["model_call_id"], refusal["code"]) == ("call3", "context_length_exceeded")
    assert refusal["created_at"] > 3.0


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


def test_finalize_honors_a_declared_terminal_response_id(tmp_path):
    result = {
        "_ng_rollout_id": "t0-r0",
        "response": {"id": "", "model": "m", "object": "response", "output": []},
        TERMINAL_RESPONSE_ID_KEY: "resp_2",
        "reward": 1.0,
    }
    built = _delivery_case(tmp_path, result)
    assert built[MASK_SAMPLE_KEY] is False
    attribution = result[TOKEN_CAPTURE_KEY]["terminal_attribution"]
    assert attribution["method"] == "declared" and attribution["chain"] == "delivered"
    assert attribution["call_id"] == "call2"


@pytest.mark.parametrize("with_aux", [False, True])
@pytest.mark.parametrize(
    ("declared_response_id", "reason"),
    [
        ("resp_unknown", "declared_terminal_not_captured"),
        ("resp_1", "witness_disagreement"),
    ],
    ids=["uncaptured-id", "conflicting-id"],
)
def test_finalize_masks_when_the_declared_response_id_is_unconfirmed(
    tmp_path: Path, with_aux: bool, declared_response_id: str, reason: str
) -> None:
    result = {
        "_ng_rollout_id": "t0-r0",
        "response": _response(
            [_assistant_item("step one"), _assistant_item("final answer")],
            response_id="resp_2",
        ),
        TERMINAL_RESPONSE_ID_KEY: declared_response_id,
        "reward": 1.0,
    }
    built = _delivery_case(tmp_path, result, with_aux=with_aux)
    # A declaration the records cannot confirm never falls back to weaker witnesses.
    assert built[MASK_SAMPLE_KEY] is True
    assert result[MASK_SAMPLE_KEY] is True
    assert not capture_build_can_retire(built)
    assert result["reward"] == 1.0
    attribution = result[TOKEN_CAPTURE_KEY]["terminal_attribution"]
    assert attribution["method"] == "none"
    assert reason in attribution["reason"]


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
