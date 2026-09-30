# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gateway transports facts and accepts framework-selected storage roots."""

import pytest

from nemo_gym.token_id_capture.external_capture import VLLMWorkerCaptureHandler
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.replay import replay_context, summarize_replay
from nemo_gym.token_id_capture.sink import (
    CaptureContext,
    register_call_intent,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.staging.records import CaptureAdmission
from tests.unit_tests.test_external_capture_handlers import _staged_coords


@pytest.fixture(params=["memory", "file"])
def ledger(request, tmp_path):
    return InMemoryLineageStore() if request.param == "memory" else FileLineageStore(tmp_path)


async def _serve(ledger, call_id, items, *, ordinary=False):
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id=call_id,
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        framework_owned_context=True,
    )
    token = set_token_sink(context)
    handler = VLLMWorkerCaptureHandler()
    try:
        await resolve_parent(items)
        candidate = context.capture_admission
        if ordinary:
            context.capture_admission = CaptureAdmission(
                **{**candidate.model_dump(exclude={"request_replay", "candidate_replay"}), "mode": "token_in"}
            )
        handler.prepare_request({"model": "model", "messages": items})
        handler.prepare_response(
            {"ng_commit_coords": _staged_coords(model_call_id=call_id, staging_key=f"rollout-1/{call_id}")}
        )
        await handler.finalize_response(
            {
                "id": f"response-{call_id}",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
                ],
            }
        )
        return context, candidate
    finally:
        reset_token_sink(token)


async def test_worker_root_clears_candidate_chain_and_keeps_source_evidence(ledger):
    await _serve(ledger, "c1", [{"role": "user", "content": "old question"}])
    context, admission = await _serve(ledger, "c2", [{"role": "user", "content": "compacted summary"}])
    assert admission.mode == "candidate"
    assert admission.parent_call_id == "c1"  # Candidate, even when content lookup misses.
    assert context.committed
    manifest = await ledger.manifest("rollout-1")
    assert not manifest["failures"]
    first, second = manifest["records"]
    assert second["parent_call_id"] is None
    assert second["mode"] == "text"
    assert second["prev_len"] == 0
    assert second["replay"]["source_digest"] != first["replay"]["source_digest"]
    # A later candidate must name only c2's new root, not its discarded parent.
    _, next_admission = await _serve(ledger, "c3", [{"role": "user", "content": "another rewrite"}])
    assert next_admission.staging_chain == ["rollout-1/c2"]


async def test_ordinary_admission_still_rejects_different_worker_parent(ledger):
    await _serve(ledger, "c1", [{"role": "user", "content": "question"}])
    context, _ = await _serve(ledger, "c2", [{"role": "user", "content": "rewrite"}], ordinary=True)
    assert not context.committed
    assert (await ledger.manifest("rollout-1"))["failures"]


async def test_candidate_lookup_failure_is_not_a_fresh_root(ledger):
    await ledger.record_failure("rollout-1", "missing", "uncommitted_call")
    with pytest.raises(ValueError, match="incomplete capture"):
        await _serve(ledger, "c2", [{"role": "user", "content": "summary"}])


def test_exposed_reasoning_survives_source_comparison():
    a = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "first"}]}]
    b = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "edited"}]}]
    assert replay_context(a) != replay_context(b)
    assert replay_context(a) != replay_context([])


def test_tool_argument_reserialization_is_qualified():
    a = {"type": "function_call", "name": "tool", "call_id": "call", "arguments": '{"x":1,"y":2}'}
    b = {**a, "arguments": '{ "y": 2, "x": 1 }'}
    assert replay_context([a]) == replay_context([b])
    assert replay_context([a]) != replay_context([{**a, "arguments": '{"x":2,"y":2}'}])
    assert summarize_replay(replay_context([a])) == summarize_replay(replay_context([b]))


def test_chat_tool_argument_reserialization_preserves_content_and_identity():
    def message(arguments, identity="call"):
        return {
            "role": "assistant",
            "tool_calls": [{"id": identity, "type": "function", "function": {"name": "tool", "arguments": arguments}}],
        }

    a = message('{"x":1,"y":2}')
    b = message('{ "y": 2, "x": 1 }')
    assert replay_context([a]) == replay_context([b])
    assert replay_context([a]) != replay_context([message('{"x":2,"y":2}')])
    assert replay_context([a]) != replay_context([message('{"x":1,"y":2}', "other")])
    assert replay_context([message('{"x":')]) != replay_context([message('{ "x":')])
    assert a["tool_calls"][0]["function"]["arguments"] == '{"x":1,"y":2}'


def test_summary_authenticates_order_roles_and_prefix_length():
    items = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    context = replay_context(items)
    summary = summarize_replay(context)
    assert summary == summarize_replay(replay_context(items + [{"role": "user", "content": "next"}]), item_count=2)
    assert summary != summarize_replay(replay_context(list(reversed(items))))
    changed_role = context.model_copy(deep=True)
    changed_role.items[0].role = "tool"
    assert summary != summarize_replay(changed_role)
    assert summary != summarize_replay(context, item_count=1)
    for invalid_count in (-1, 3):
        with pytest.raises(ValueError, match="prefix length"):
            summarize_replay(context, item_count=invalid_count)


async def test_intent_is_atomic_and_unknown_ack_is_retained(ledger):
    await ledger.begin_call("rollout-1", "c1", None)
    manifest = await ledger.manifest("rollout-1")
    assert manifest["attempted_call_ids"] == ["c1"]
    assert manifest["pending_call_ids"] == ["c1"]
    with pytest.raises(ValueError, match="pending"):
        await ledger.begin_call("rollout-1", "c2", None)
    await ledger.record_failure("rollout-1", "c1", "request_finished_without_staged_coordinates")
    assert (await ledger.manifest("rollout-1"))["pending_call_ids"] == ["c1"]


async def test_intent_rejects_stale_candidate_and_commit_resolves_it(ledger):
    await ledger.begin_call("rollout-1", "c1", None)
    # The HTTP admission normally precedes begin_call; finish this already admitted call directly.
    from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit

    coords = _staged_coords(model_call_id="c1", staging_key="rollout-1/c1")
    fields = {name: value for name, value in coords.items() if name in CallRecord.model_fields}
    fields.update(
        mode="text",
        response_id="response-c1",
        replay=summarize_replay(replay_context([{"role": "user", "content": "hello"}], render_digest="a" * 64)),
    )
    await ledger.record(
        CaptureLedgerCommit(rollout_id="rollout-1", record=CallRecord(**fields), request_items=[], response_items=[])
    )
    assert not (await ledger.manifest("rollout-1"))["pending_call_ids"]
    with pytest.raises(ValueError, match="Ledger changed"):
        await ledger.begin_call("rollout-1", "c2", None)
    await ledger.begin_call("rollout-1", "c2", "c1")
    assert (await ledger.manifest("rollout-1"))["pending_call_ids"] == ["c2"]


@pytest.mark.parametrize("raced", [False, True])
async def test_selected_candidate_and_atomic_ledger_head_are_independent(ledger, raced):
    items = [{"role": "user", "content": "question"}]
    await _serve(ledger, "accepted", items)
    await _serve(ledger, "rejected", items)
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id="next",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        framework_owned_context=True,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(items, parent_response_id="response-accepted")
        assert context.capture_admission.parent_call_id == "accepted"
        assert context.capture_admission.staging_chain == ["rollout-1/accepted"]
        assert context.admitted_latest_call_id == "rejected"
        if raced:
            await _serve(ledger, "racing-commit", items)
            with pytest.raises(ValueError, match="Ledger changed"):
                await register_call_intent()
        else:
            await register_call_intent()
            assert (await ledger.manifest("rollout-1"))["pending_call_ids"] == ["next"]
    finally:
        reset_token_sink(token)


async def test_selected_response_must_exist_uniquely_in_this_attempt(ledger):
    await _serve(ledger, "accepted", [{"role": "user", "content": "question"}])
    context = CaptureContext(
        rollout_id="other-attempt",
        model_call_id="next",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        framework_owned_context=True,
    )
    token = set_token_sink(context)
    try:
        with pytest.raises(ValueError, match="missing or ambiguous"):
            await resolve_parent([], parent_response_id="response-accepted")
        assert context.capture_admission is None
    finally:
        reset_token_sink(token)


@pytest.mark.parametrize("ambiguous", [False, True])
async def test_empty_tool_content_candidate_is_not_just_latest_call(ledger, ambiguous):
    history = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call", "type": "function", "function": {"name": "check", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call", "content": "checked"},
    ]
    await _serve(ledger, "accepted", history)
    await _serve(ledger, "other", history if ambiguous else [{"role": "user", "content": "other task"}])
    echoed = [
        history[0],
        {**history[1], "content": ""},
        history[2],
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        {"role": "user", "content": "continue"},
    ]
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id="next",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        framework_owned_context=True,
    )
    token = set_token_sink(context)
    try:
        if ambiguous:
            with pytest.raises(ValueError, match="Ambiguous source-prefix candidate"):
                await resolve_parent(echoed)
        else:
            await resolve_parent(echoed)
            assert context.capture_admission.parent_call_id == "accepted"
            assert context.admitted_latest_call_id == "other"
    finally:
        reset_token_sink(token)
