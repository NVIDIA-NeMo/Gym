# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gateway selects source-compatible parents and preserves root capture."""

import asyncio

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


async def _serve(ledger, call_id, items, *, ordinary=False, splice=False, started=None, finish=None):
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
        await register_call_intent()
        if ordinary:
            context.framework_owned_context = False
            context.capture_admission = CaptureAdmission(
                **{
                    **candidate.model_dump(
                        exclude={"request_replay", "candidate_replay", "requires_empty_tool_content_equivalence"}
                    ),
                    "mode": "token_in",
                }
            )
        handler.prepare_request({"model": "model", "messages": items})
        if started is not None:
            started.set()
        if finish is not None:
            await finish.wait()
        coords = dict(model_call_id=call_id, staging_key=f"rollout-1/{call_id}")
        if splice:
            coords.update(
                parent_call_id=candidate.parent_call_id, prev_len=candidate.prev_len, cum_len=candidate.prev_len + 3
            )
        handler.prepare_response({"ng_commit_coords": _staged_coords(**coords)})
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


async def test_source_rewrite_roots_without_candidate_and_keeps_source_evidence(ledger):
    await _serve(ledger, "c1", [{"role": "user", "content": "old question"}])
    context, admission = await _serve(ledger, "c2", [{"role": "user", "content": "compacted summary"}])
    assert admission.mode == "text"
    assert admission.parent_call_id is None
    assert admission.staging_chain == []
    assert context.committed
    manifest = await ledger.manifest("rollout-1")
    assert not manifest["failures"]
    first, second = manifest["records"]
    assert second["parent_call_id"] is None
    assert second["mode"] == "text"
    assert second["prev_len"] == 0
    assert second["replay"]["source_digest"] != first["replay"]["source_digest"]
    # A continuation must name only c2's new root.
    _, next_admission = await _serve(
        ledger,
        "c3",
        [
            {"role": "user", "content": "compacted summary"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            {"role": "user", "content": "continue"},
        ],
        splice=True,
    )
    assert next_admission.staging_chain == ["rollout-1/c2"]


async def test_ordinary_admission_still_rejects_different_worker_parent(ledger):
    await _serve(ledger, "c1", [{"role": "user", "content": "question"}])
    context, _ = await _serve(
        ledger,
        "c2",
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            {"role": "user", "content": "continue"},
        ],
        ordinary=True,
    )
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
    await ledger.begin_call("rollout-1", "c1")
    manifest = await ledger.manifest("rollout-1")
    assert manifest["attempted_call_ids"] == ["c1"]
    assert manifest["pending_call_ids"] == ["c1"]
    await ledger.begin_call("rollout-1", "c2")
    with pytest.raises(ValueError, match="already registered"):
        await ledger.begin_call("rollout-1", "c2")
    await ledger.record_failure("rollout-1", "c1", "request_finished_without_staged_coordinates")
    assert (await ledger.manifest("rollout-1"))["pending_call_ids"] == ["c1", "c2"]


async def test_commit_resolves_its_intent(ledger):
    await ledger.begin_call("rollout-1", "c1")
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
    await ledger.begin_call("rollout-1", "c2")
    assert (await ledger.manifest("rollout-1"))["pending_call_ids"] == ["c2"]


@pytest.mark.parametrize("raced", [False, True])
async def test_selected_candidate_survives_an_unrelated_commit(ledger, raced):
    items = [{"role": "user", "content": "question"}]
    await _serve(ledger, "accepted", items)
    await _serve(ledger, "rejected", [{"role": "user", "content": "another branch"}])
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
        await resolve_parent(
            items
            + [
                {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
                {"role": "user", "content": "next"},
            ]
        )
        assert context.capture_admission.parent_call_id == "accepted"
        assert context.capture_admission.staging_chain == ["rollout-1/accepted"]
        if raced:
            await _serve(ledger, "racing-commit", items)
        await register_call_intent()
        assert (await ledger.manifest("rollout-1"))["pending_call_ids"] == ["next"]
    finally:
        reset_token_sink(token)


async def test_candidate_lookup_is_scoped_to_this_attempt(ledger):
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
        await resolve_parent([{"role": "user", "content": "question"}])
        assert context.capture_admission.parent_call_id is None
        assert context.capture_admission.staging_chain == []
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
            assert context.capture_admission.requires_empty_tool_content_equivalence is True
    finally:
        reset_token_sink(token)


@pytest.mark.parametrize("first", ["b", "c"])
@pytest.mark.parametrize("splice", [False, True])
async def test_concurrent_calls_finish_after_admissions_close(ledger, first, splice):
    history = [{"role": "user", "content": "task"}]
    await _serve(ledger, "a", history)
    continuation = history + [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        {"role": "user", "content": "continue"},
    ]
    # c either shares a or rewrites its context into a separate root.
    prompts = {"b": continuation, "c": continuation if splice else [{"role": "user", "content": "summary"}]}
    starts = {name: asyncio.Event() for name in prompts}
    finishes = {name: asyncio.Event() for name in prompts}
    tasks = {
        name: asyncio.create_task(
            _serve(ledger, name, prompt, splice=(name == "b" or splice), started=starts[name], finish=finishes[name])
        )
        for name, prompt in prompts.items()
    }
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in starts.values())), timeout=5)
        closed = await ledger.close_admissions("rollout-1")
        assert set(closed["pending_call_ids"]) == {"b", "c"}
        assert await ledger.close_admissions("rollout-1") == closed
        with pytest.raises(ValueError, match="closed"):
            await ledger.begin_call("rollout-1", "late")
        for name in (first, "c" if first == "b" else "b"):
            finishes[name].set()
            await tasks[name]
        manifest = await ledger.manifest("rollout-1")
        assert not manifest["pending_call_ids"] and not manifest["failures"]
        by_call = {r["model_call_id"]: r for r in manifest["records"]}
        assert by_call["b"]["parent_call_id"] == "a"
        assert by_call["b"]["cum_len"] == 6
        assert by_call["c"]["parent_call_id"] == ("a" if splice else None)
        assert [r["model_call_id"] for r in manifest["records"]][1] == first
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


async def test_continuation_rebuilds_full_staging_chain(ledger):
    history = [{"role": "user", "content": "task"}]
    await _serve(ledger, "a", history)
    for name, expected in (("b", ["rollout-1/a"]), ("c", ["rollout-1/a", "rollout-1/b"])):
        history += [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            {"role": "user", "content": "more"},
        ]
        _, candidate = await _serve(ledger, name, history, splice=True)
        assert candidate.staging_chain == expected


async def test_close_racing_begin_has_no_unaccounted_dispatch(ledger):
    # Use separate instances for the file store, as separate serving processes do.
    peer = FileLineageStore(ledger._ledger_root) if isinstance(ledger, FileLineageStore) else ledger
    for index in range(10):
        owner = f"race-{index}"
        begin, snapshot = await asyncio.gather(
            peer.begin_call(owner, "call"), ledger.close_admissions(owner), return_exceptions=True
        )
        assert isinstance(snapshot, dict)
        if isinstance(begin, ValueError):
            assert "closed" in str(begin)
            assert snapshot["attempted_call_ids"] == []
        else:
            assert begin is None
            assert snapshot["pending_call_ids"] == ["call"]
        with pytest.raises(ValueError, match="closed"):
            await peer.begin_call(owner, "late")


@pytest.mark.parametrize(
    "edit",
    [
        "exact",
        "null_empty",
        "empty_null",
        "empty_user",
        "empty_assistant",
        "empty_tool",
        "last_response",
        "question",
        "reasoning",
        "media",
        "shortened",
        "order",
        "no_suffix",
        "assistant_suffix",
        "tool_id",
        "arguments",
        "text",
        "whitespace",
        "render",
    ],
)
async def test_source_and_effective_options_choose_root_or_proven_candidate(ledger, edit):
    from copy import deepcopy

    history = [
        {"role": "user", "content": [{"type": "input_image", "image_url": "https://example.test/a.png"}]},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "exposed reasoning",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "tool", "arguments": '{"x":1}'}}],
        },
        {"role": "tool", "content": "observation", "tool_call_id": "call-1"},
    ]
    empty_index = {"empty_user": 0, "empty_assistant": 1, "empty_tool": 2}.get(edit)
    if empty_index is not None:
        history[empty_index]["content"] = None
        history[empty_index].pop("tool_calls", None)
    if edit == "empty_null":
        history[1]["content"] = ""
    await _serve(ledger, "a", history)
    current = deepcopy(history) + [
        {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        {"role": "user", "content": "continue"},
    ]
    if edit == "null_empty":
        current[1]["content"] = ""
    elif edit == "empty_null":
        current[1]["content"] = None
    elif empty_index is not None:
        current[empty_index]["content"] = ""
    elif edit == "last_response":
        current[-2]["content"] = "changed answer"
    elif edit == "question":
        current[0]["content"] = "new question"
    elif edit == "reasoning":
        current[1].pop("reasoning")
    elif edit == "media":
        current[0]["content"][0]["image_url"] = "https://example.test/b.png"
    elif edit == "shortened":
        current = current[:1]
    elif edit == "order":
        current[:2] = reversed(current[:2])
    elif edit == "no_suffix":
        current.pop()
    elif edit == "assistant_suffix":
        current[-1]["role"] = "assistant"
    elif edit == "tool_id":
        current[1]["tool_calls"][0]["id"] = "other"
    elif edit == "arguments":
        current[1]["tool_calls"][0]["function"]["arguments"] = '{"x":2}'
    elif edit in ("text", "whitespace"):
        current[1]["content"] = "new text" if edit == "text" else " "
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id="b",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        framework_owned_context=True,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(current)
        payload = VLLMWorkerCaptureHandler().prepare_request(
            {
                "model": "different" if edit == "render" else "model",
                "messages": current,
                "required_prefix_token_ids": [999],
            }
        )
        admission = context.capture_admission
        matched = edit in ("exact", "null_empty", "empty_null")
        assert admission.mode == ("candidate" if matched else "text")
        assert admission.parent_call_id == ("a" if matched else None)
        assert admission.requires_empty_tool_content_equivalence == (
            (edit in ("null_empty", "empty_null")) if matched else None
        )
        assert admission.staging_chain == (["rollout-1/a"] if matched else [])
        assert context.parent_staging_chain == admission.staging_chain
        assert admission.request_replay.render_digest is not None
        assert "required_prefix_token_ids" not in payload
    finally:
        reset_token_sink(token)


@pytest.mark.parametrize("option", ["truncate_prompt_tokens", "continue_final_message"])
async def test_framework_direct_root_keeps_unsupported_option_guard(ledger, option):
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id="root",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        framework_owned_context=True,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent([{"role": "user", "content": "fresh"}])
        assert context.capture_admission.mode == "text"
        with pytest.raises(ValueError, match="without truncation or assistant prefill"):
            VLLMWorkerCaptureHandler().prepare_request({"model": "model", option: 1})
    finally:
        reset_token_sink(token)


@pytest.mark.parametrize("mode", ["text", "token_in"])
def test_legacy_admission_wire_omits_new_candidate_proof(mode):
    fields = (
        {}
        if mode == "text"
        else dict(parent_call_id="a", prev_len=1, required_prefix_token_ids=[1], parent_chain_hash="a" * 64)
    )
    admission = CaptureAdmission(rollout_id="r", model_call_id="b", mode=mode, **fields)
    wire = VLLMWorkerCaptureHandler()._prepare_admitted_request({}, admission)["ng_capture"]
    assert "requires_empty_tool_content_equivalence" not in wire
    assert CaptureAdmission.model_validate(wire) == admission


@pytest.mark.parametrize("continuation", [False, True])
async def test_framework_ack_cannot_change_the_admitted_parent(ledger, continuation):
    items = [{"role": "user", "content": "question"}]
    await _serve(ledger, "a", items)
    if continuation:
        items += [
            {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            {"role": "user", "content": "next"},
        ]
    context = CaptureContext(
        rollout_id="rollout-1",
        model_call_id="b",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        framework_owned_context=True,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(items)
        await register_call_intent()
        handler = VLLMWorkerCaptureHandler()
        handler.prepare_request({"model": "model", "messages": items})
        handler.prepare_response(
            {
                "ng_commit_coords": _staged_coords(
                    model_call_id="b",
                    parent_call_id="unrelated",
                    prev_len=3,
                    cum_len=6,
                    staging_key="rollout-1/b",
                )
            }
        )
        await handler.finalize_response({"id": "response-b", "output": []})
        manifest = await ledger.manifest("rollout-1")
        assert not context.committed and manifest["failures"]
        assert [record["model_call_id"] for record in manifest["records"]] == ["a"]
    finally:
        reset_token_sink(token)
