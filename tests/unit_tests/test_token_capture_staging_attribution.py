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
"""Test manifest-level terminal attribution: the witnesses join on token-free rows.

The ``/run`` result's ``response`` is the object the verifier scored.
Attribution joins it to exactly one ``CallRecord`` through the served
response id, the recorded content fingerprints, or an explicit declaration.
Every abstention and disagreement fails closed; the caller's parent-link
policy (``select_terminal_call``) remains the no-witness fallback.
"""

import hashlib

import pydantic
import pytest

from nemo_gym.token_id_capture.fingerprint import FINGERPRINT_VERSION, assistant_fingerprint
from nemo_gym.token_id_capture.staging import resolve_terminal
from nemo_gym.token_id_capture.staging.records import CallRecord


def _assistant_item(text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _row(
    call_id: str,
    *,
    parent: str | None = None,
    prev_len: int = 0,
    delta_len: int = 100,
    response_id: str,
    text: str | None = None,
    request_texts: list[str] | None = None,
    cumulative_hash: str | None = None,
    chain_hash: str | None = None,
    fingerprint_version: int = FINGERPRINT_VERSION,
) -> CallRecord:
    output_fingerprint = assistant_fingerprint([_assistant_item(text)]) if text is not None else None
    continuation_fingerprint = (
        assistant_fingerprint([_assistant_item(t) for t in (request_texts or [])] + [_assistant_item(text)])
        if text is not None
        else None
    )
    return CallRecord(
        model_call_id=call_id,
        parent_call_id=parent,
        prev_len=prev_len,
        delta_len=delta_len,
        cum_len=prev_len + delta_len,
        weight_version=3,
        digest="a" * 64,
        extras_digest="b" * 64,
        staging_key=f"r0/{call_id}",
        mode="text" if parent is None else "token_in",
        # Distinct per call unless a test deliberately makes rows collide.
        chain_hash=chain_hash or hashlib.sha256(f"chain:{call_id}".encode()).hexdigest(),
        cumulative_hash=cumulative_hash or hashlib.sha256(f"cum:{call_id}".encode()).hexdigest(),
        response_id=response_id,
        output_fingerprint=output_fingerprint or None,
        continuation_fingerprint=continuation_fingerprint or None,
        fingerprint_version=fingerprint_version if text is not None else 0,
    )


def _chain() -> list[CallRecord]:
    return [
        _row("c1", response_id="chatcmpl-1", text="step one", cumulative_hash="1" * 64),
        _row(
            "c2",
            parent="c1",
            prev_len=100,
            response_id="chatcmpl-2",
            text="final answer",
            request_texts=["step one"],
            cumulative_hash="2" * 64,
        ),
    ]


def _response(items: list[dict], response_id: str = "") -> dict:
    return {"id": response_id, "model": "m", "object": "response", "output": items}


# --- schema strictness ----------------------------------------------------------


def test_a_custody_row_without_a_response_id_fails_validation():
    with pytest.raises(pydantic.ValidationError):
        CallRecord(
            model_call_id="c1",
            prev_len=0,
            delta_len=1,
            cum_len=1,
            weight_version=0,
            digest="a" * 64,
            extras_digest="b" * 64,
            staging_key="r0/c1",
            mode="text",
        )


# --- declared witness -----------------------------------------------------------


def test_declared_response_id_attributes():
    att = resolve_terminal(_chain(), None, declared_response_id="chatcmpl-2")
    assert att.attributed and att.model_call_id == "c2" and att.method == "declared"


def test_declared_miss_masks_and_never_falls_back():
    # The scored response would attribute c2, but the declaration is
    # authoritative: a miss is a defect, not an invitation to guess.
    response = _response([_assistant_item("final answer")], response_id="chatcmpl-2")
    att = resolve_terminal(_chain(), response, declared_response_id="chatcmpl-ghost")
    assert not att.attributed
    assert "declared_terminal_not_captured" in att.reason


def test_declared_corroborated_by_response_id_and_content():
    response = _response([_assistant_item("final answer")], response_id="chatcmpl-2")
    att = resolve_terminal(_chain(), response, declared_response_id="chatcmpl-2")
    assert att.attributed and att.model_call_id == "c2" and att.method == "declared"
    assert "corroborated_by=response_id+content" in att.reason


# --- response-id witness --------------------------------------------------------


def test_response_id_attributes_without_a_declaration():
    response = _response([], response_id="chatcmpl-2")
    att = resolve_terminal(_chain(), response)
    assert att.attributed and att.model_call_id == "c2" and att.method == "response_id"


def test_response_id_resolves_a_divergent_final_retry():
    # Same prompt, different generations: content cannot say which was kept,
    # but possession of the served id can.
    records = [
        _row("c2a", parent="c1", prev_len=100, response_id="chatcmpl-2a", text="answer A", cumulative_hash="a" * 64),
        _row("c2b", parent="c1", prev_len=100, response_id="chatcmpl-2b", text="answer B", cumulative_hash="c" * 64),
        _row("c1", response_id="chatcmpl-1", text="step", cumulative_hash="1" * 64),
    ]
    att = resolve_terminal(records, _response([_assistant_item("answer B")], response_id="chatcmpl-2b"))
    assert att.attributed and att.model_call_id == "c2b"


def test_duplicated_response_id_abstains_but_content_still_attributes():
    # Backend id reuse across different generations is a defect; the id
    # witness abstains and leaves the trail while content attributes.
    records = [
        _row("ca", response_id="chatcmpl-dup", text="answer A", cumulative_hash="a" * 64),
        _row("cb", response_id="chatcmpl-dup", text="answer B", cumulative_hash="c" * 64),
    ]
    att = resolve_terminal(records, _response([_assistant_item("answer B")], response_id="chatcmpl-dup"))
    assert att.attributed and att.model_call_id == "cb" and att.method == "content"
    assert "response_id_ambiguous" in att.reason


def test_identical_retries_collapse_to_one_row():
    # Two servings with identical staged sequences are interchangeable for
    # training; the smallest call id wins deterministically.
    records = [
        _row("cb", response_id="chatcmpl-x", text="same", cumulative_hash="d" * 64, chain_hash="e" * 64),
        _row("ca", response_id="chatcmpl-x", text="same", cumulative_hash="d" * 64, chain_hash="e" * 64),
    ]
    att = resolve_terminal(records, _response([], response_id="chatcmpl-x"))
    assert att.attributed and att.model_call_id == "ca" and att.method == "response_id"


# --- content witness ------------------------------------------------------------


def test_trailing_block_attributes_a_merged_transcript_without_ids():
    response = _response(
        [
            _assistant_item("step one"),
            {"type": "function_call_output", "call_id": "t", "output": "ok"},
            _assistant_item("final answer"),
        ]
    )
    att = resolve_terminal(_chain(), response)
    assert att.attributed and att.model_call_id == "c2" and att.method == "content"
    assert "response_has_no_id" in att.reason


def test_continuation_fingerprint_attributes_a_full_transcript():
    # A merged transcript of every model turn matches only the cumulative
    # reading recorded on the terminal row.
    response = _response([_assistant_item("step one"), _assistant_item("final answer")])
    att = resolve_terminal(_chain(), response)
    assert att.attributed and att.model_call_id == "c2" and att.method == "content"


def test_final_turn_only_response_matches_own_output():
    att = resolve_terminal(_chain(), _response([_assistant_item("final answer")]))
    assert att.attributed and att.model_call_id == "c2" and att.method == "content"


def test_transcript_with_a_pending_tool_result_matches_the_cumulative_reading():
    # The model-authored spine of the transcript equals c2's request + output,
    # so the recorded continuation fingerprint attributes the last *completed*
    # call even though the rollout truncated on a pending tool result.
    response = _response(
        [
            _assistant_item("step one"),
            _assistant_item("final answer"),
            {"type": "function_call_output", "call_id": "t", "output": "ok"},
        ]
    )
    att = resolve_terminal(_chain(), response)
    assert att.attributed and att.model_call_id == "c2" and att.method == "content"


def test_pending_tool_result_skips_the_trailing_reading_without_cumulative_keys():
    # Rows without continuation fingerprints leave only the trailing-block
    # reading, which must not fire when the transcript ends in a non-model
    # item: matching a mid-chain call would be wrong.
    records = []
    for record in _chain():
        records.append(record.model_copy(update={"continuation_fingerprint": None}))
    response = _response(
        [
            _assistant_item("step one"),
            _assistant_item("final answer"),
            {"type": "function_call_output", "call_id": "t", "output": "ok"},
        ]
    )
    att = resolve_terminal(records, response)
    assert not att.attributed and "no_content_match" in att.reason


def test_a_mutated_echo_matches_nothing():
    att = resolve_terminal(_chain(), _response([_assistant_item("final answer [redacted]")]))
    assert not att.attributed and "no_content_match" in att.reason


def test_repeated_identical_output_at_different_depths_abstains():
    records = [
        _row("c1", response_id="chatcmpl-1", text="done", cumulative_hash="1" * 64),
        _row("c2", parent="c1", prev_len=100, response_id="chatcmpl-2", text="done", cumulative_hash="2" * 64),
    ]
    response = _response(
        [
            _assistant_item("done"),
            {"type": "function_call_output", "call_id": "t", "output": "ok"},
            _assistant_item("done"),
        ]
    )
    att = resolve_terminal(records, response)
    assert not att.attributed and "content_ambiguous" in att.reason


def test_version_mismatched_fingerprints_never_match():
    records = [
        _row("c1", response_id="chatcmpl-1", text="final answer", fingerprint_version=99, cumulative_hash="1" * 64),
    ]
    att = resolve_terminal(records, _response([_assistant_item("final answer")]))
    assert not att.attributed and "no_content_match" in att.reason


# --- corroboration and fallback --------------------------------------------------


def test_witness_disagreement_fails_closed():
    # The scored id names c1; the content names c2. Contradiction attributes
    # nothing and persists the disagreement.
    response = _response([_assistant_item("final answer")], response_id="chatcmpl-1")
    att = resolve_terminal(_chain(), response)
    assert not att.attributed
    assert "witness_disagreement[" in att.reason


def test_no_response_object_abstains():
    att = resolve_terminal(_chain(), None)
    assert not att.attributed and "no_response_object" in att.reason


def test_no_witness_reports_every_abstention():
    att = resolve_terminal(_chain(), _response([], response_id="chatcmpl-ghost"))
    assert not att.attributed
    assert "response_id_no_match" in att.reason
    assert "response_has_no_output" in att.reason
