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
"""Scoping of parent search, terminal attribution and outcomes by agent instance and model server.

A lineage scope is the calls of one agent instance on one model server
(``nemo_gym.models.token_capture.lineage_scope``).
"""

import json

import pytest

from nemo_gym.models.token_capture.consumer import (
    AGENT_INSTANCE_BUILDS_KEY,
    trajectories_for_rollout,
    trajectories_from_source,
)
from nemo_gym.models.token_capture.external_capture import VLLMWorkerCaptureHandler
from nemo_gym.models.token_capture.lineage import (
    FIRST_CALL_REASON,
    MODEL_SERVER_CHANGED_REASON,
    FileLineageStore,
    InMemoryLineageStore,
    stamp_continuation,
)
from nemo_gym.models.token_capture.lineage_scope import primary_agent_instance
from nemo_gym.models.token_capture.protocols import RolloutRetiredError
from nemo_gym.models.token_capture.records import (
    CallIdentity,
    ModelRef,
    ParentResolutionStatus,
    TokenEntry,
    stamp_lineage,
)
from nemo_gym.models.token_capture.sink import (
    CaptureContext,
    register_call_intent,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.models.token_capture.staging.records import RolloutManifest
from nemo_gym.models.token_capture.store import TokenCaptureStore
from tests.unit_tests.models.test_external_capture_handlers import _staged_coords


ROLLOUT = "t0-r0"
MODEL_SERVER_A = ModelRef(name="model_server_a")
MODEL_SERVER_B = ModelRef(name="model_server_b")
UNNAMED = CallIdentity(model_ref=MODEL_SERVER_A)
ASSISTANT = CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_A)
JUDGE = CallIdentity(agent_instance="judge", model_ref=MODEL_SERVER_A)
USER_SIM = CallIdentity(agent_instance="user", model_ref=MODEL_SERVER_A)

USER = {"role": "user", "content": "solve the task"}
ANSWER = {"role": "assistant", "content": "the answer is 4"}
JUDGE_PROMPT = {"role": "user", "content": "Grade the transcript above."}


def _entry(
    model_call_id: str,
    identity: CallIdentity,
    request_items: list[dict],
    output_text: str,
    *,
    prompt: list[int],
    generation: list[int],
    parent: str | None = None,
    response_id: str | None = None,
) -> TokenEntry:
    entry = TokenEntry(
        capture_key=ROLLOUT,
        model_call_id=model_call_id,
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        generation_log_probs=[-0.1] * len(generation),
        output_items=[{"type": "message", "role": "assistant", "content": output_text}],
        response_id=response_id or f"resp_{model_call_id}",
        agent_instance=identity.agent_instance,
        agent_kind=identity.agent_kind,
        model_ref=identity.model_ref,
    )
    status = ParentResolutionStatus.RESOLVED if parent else ParentResolutionStatus.ROOT
    stamp_lineage(entry, parent, parent_resolution=status)
    return stamp_continuation(entry, request_items)


def _assistant_call() -> TokenEntry:
    return _entry("a1", ASSISTANT, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])


def _judge_call() -> TokenEntry:
    return _entry("j1", JUDGE, [USER, ANSWER, JUDGE_PROMPT], "4/5", prompt=[7, 8], generation=[9])


def _store_with(tmp_path, entries, *, intents=(), marks=()) -> TokenCaptureStore:
    store = TokenCaptureStore(tmp_path)
    for model_call_id, identity in intents:
        store._begin_call(ROLLOUT, model_call_id, identity)
    for entry in entries:
        store.append(entry)
    for model_call_id, identity in marks:
        store._mark_incomplete(ROLLOUT, model_call_id, identity)
    return store


def _generation_tokens(built: dict) -> list[list[int]]:
    return [item["generation_token_ids"] for item in built["rebuilt_response"]["output"]]


# --- parent search ---


@pytest.fixture(params=["memory", "file"])
def lineage(request, tmp_path):
    if request.param == "memory":
        store = InMemoryLineageStore()

        async def commit(entry: TokenEntry) -> None:
            await store.put(entry)

    else:
        token_store = TokenCaptureStore(tmp_path)
        store = FileLineageStore(tmp_path)

        async def commit(entry: TokenEntry) -> None:
            token_store.append(entry)

    return store, commit


@pytest.fixture(params=["memory", "file"])
def ledger(request, tmp_path):
    return InMemoryLineageStore() if request.param == "memory" else FileLineageStore(tmp_path)


REPLAYED = [USER, ANSWER, JUDGE_PROMPT]


async def test_a_judge_s_first_request_replaying_the_assistant_s_turns_is_a_root(lineage, tmp_path_factory):
    # The judge's lineage scope has made no call, so the replayed turns are input, not the judge's earlier output.
    store, commit = lineage
    await commit(_assistant_call())
    transcript_as_text = [{"role": "user", "content": "Grade this transcript: solve the task / the answer is 4"}]

    judge = await store.resolve(ROLLOUT, REPLAYED, identity=JUDGE)
    as_text = await store.resolve(ROLLOUT, transcript_as_text, identity=JUDGE)
    assistant = await store.resolve(ROLLOUT, REPLAYED, identity=ASSISTANT)

    assert (judge.status, judge.reason, judge.match) == (ParentResolutionStatus.ROOT, FIRST_CALL_REASON, None)
    assert (as_text.status, as_text.reason) == (ParentResolutionStatus.ROOT, "")
    assert assistant.status == ParentResolutionStatus.RESOLVED and assistant.match.model_call_id == "a1"
    judge_call = _judge_call()
    stamp_lineage(judge_call, None, parent_resolution=judge.status)
    directory = tmp_path_factory.mktemp("build")
    _store_with(directory, [_assistant_call(), judge_call])
    built = trajectories_for_rollout(ROLLOUT, [directory])
    assert built["mask_sample"] is False and _generation_tokens(built) == [[10, 11]]
    judge_build = built[AGENT_INSTANCE_BUILDS_KEY]["judge"]
    assert judge_build["mask_sample"] is False and _generation_tokens(judge_build) == [[9]]


@pytest.mark.parametrize("external_staging", [False, True])
async def test_local_capture_and_worker_custody_agree_on_a_judge_s_first_request(tmp_path, external_staging):
    TokenCaptureStore(tmp_path).append(_assistant_call())
    context = CaptureContext(
        capture_key=ROLLOUT,
        model_call_id="j1",
        token_sink=None,
        lineage_store=FileLineageStore(tmp_path),
        external_staging=external_staging,
        agent_instance=JUDGE.agent_instance,
        model_ref=JUDGE.model_ref,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(REPLAYED)
    finally:
        reset_token_sink(token)

    assert context.parent_resolution.status == ParentResolutionStatus.ROOT
    if external_staging:
        assert context.capture_admission.mode == "text" and context.capture_admission.parent_call_id is None


async def test_a_request_with_unmatched_model_turns_is_unresolved_once_its_lineage_scope_has_called(lineage):
    store, commit = lineage
    await commit(_assistant_call())
    await commit(_judge_call())
    rewritten = [USER, {"role": "assistant", "content": "a rewritten answer"}, JUDGE_PROMPT]

    resolution = await store.resolve(ROLLOUT, rewritten, identity=JUDGE)

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.UNRESOLVED, "no_match")


async def test_a_lineage_scope_whose_first_call_was_lost_does_not_start_over_as_a_root(tmp_path):
    # The lost call left its intent. The next request may continue it, so it is not the lineage scope's first call.
    token_store = TokenCaptureStore(tmp_path)
    token_store.append(_assistant_call())
    token_store._begin_call(ROLLOUT, "j0", JUDGE)

    resolution = await FileLineageStore(tmp_path).resolve(ROLLOUT, REPLAYED, identity=JUDGE)

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.UNRESOLVED, "no_match")


async def test_a_lineage_scope_with_only_a_failed_custody_row_does_not_start_over_as_a_root(ledger):
    await ledger.record_failure(ROLLOUT, "a1", "worker_capture_failed", identity=ASSISTANT)
    await ledger.record_failure(ROLLOUT, "j0", "worker_capture_failed", identity=JUDGE)

    judge = await ledger.resolve(ROLLOUT, REPLAYED, identity=JUDGE)
    user = await ledger.resolve(ROLLOUT, REPLAYED, identity=USER_SIM)

    assert (judge.status, judge.reason) == (ParentResolutionStatus.UNRESOLVED, "no_match")
    assert (user.status, user.reason) == (ParentResolutionStatus.ROOT, FIRST_CALL_REASON)


async def test_the_memory_index_keeps_a_request_unresolved_when_it_holds_no_call_of_the_rollout():
    # An evicted rollout looks the same as one that made no call, so the memory-only index cannot call it a root.
    resolution = await InMemoryLineageStore().resolve(ROLLOUT, REPLAYED, identity=JUDGE)

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.UNRESOLVED, "no_match")


async def test_a_file_store_roots_the_first_request_of_a_rollout_that_carries_model_turns(tmp_path):
    resolution = await FileLineageStore(tmp_path).resolve(ROLLOUT, REPLAYED, identity=UNNAMED)

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.ROOT, FIRST_CALL_REASON)


async def test_a_continuation_on_another_model_server_is_a_root(lineage):
    store, commit = lineage
    await commit(_assistant_call())
    moved = CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_B)

    resolution = await store.resolve(ROLLOUT, [USER, ANSWER, {"role": "user", "content": "continue"}], identity=moved)

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.ROOT, MODEL_SERVER_CHANGED_REASON)


async def test_another_agent_instance_s_call_on_another_model_server_is_not_a_model_server_change(lineage):
    store, commit = lineage
    await commit(_assistant_call())

    resolution = await store.resolve(
        ROLLOUT, REPLAYED, identity=CallIdentity(agent_instance="judge", model_ref=MODEL_SERVER_B)
    )

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.ROOT, FIRST_CALL_REASON)


async def test_a_request_no_model_server_served_is_not_a_model_server_change(lineage):
    store, commit = lineage
    await commit(_assistant_call())
    never_served = [USER, {"role": "assistant", "content": "nobody served this"}, JUDGE_PROMPT]

    resolution = await store.resolve(
        ROLLOUT, never_served, identity=CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_B)
    )

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.ROOT, FIRST_CALL_REASON)


async def test_an_unnamed_request_does_not_see_named_calls(lineage):
    store, commit = lineage
    await commit(_assistant_call())

    resolution = await store.resolve(ROLLOUT, REPLAYED, identity=UNNAMED)

    assert resolution.match is None and resolution.status != ParentResolutionStatus.RESOLVED


async def test_retire_leaves_no_lineage_scope_in_the_lineage_caches(lineage):
    store, commit = lineage
    materialized_before = getattr(store, "_materialized_tokens", 0)
    await commit(_assistant_call())
    await commit(_judge_call())
    await store.resolve(ROLLOUT, REPLAYED, identity=ASSISTANT)

    if isinstance(store, FileLineageStore):
        # Resolving cached the assistant's parent tokens, which count against the global token bound.
        assert store._materialized_tokens > materialized_before
    await store.retire([ROLLOUT])

    if isinstance(store, InMemoryLineageStore):
        assert len(store.index) == 0
    else:
        assert ROLLOUT not in store._cache and ROLLOUT not in store._materialized
        assert store._materialized_tokens == materialized_before


# --- a retired capture key ---


async def test_a_lineage_scope_s_first_call_is_refused_on_a_retired_capture_key_that_still_has_rows(ledger):
    # ``retire`` marks the capture key before it removes the ledger, so another scope's rows can still be there.
    await ledger.record_failure(ROLLOUT, "a1", "worker_capture_failed", identity=ASSISTANT)
    if isinstance(ledger, InMemoryLineageStore):
        ledger._retired.add(ROLLOUT)
    else:
        ledger._retired_path(ROLLOUT).touch()

    with pytest.raises(RolloutRetiredError):
        await ledger.has_rows(ROLLOUT, identity=JUDGE)
    assert await ledger.has_rows(ROLLOUT, identity=ASSISTANT)


# --- source-history admission (worker custody) ---


async def _serve(ledger, call_id: str, items: list[dict], identity: CallIdentity):
    """Admit and commit one call by source history, as the vLLM worker capture handler does."""
    context = CaptureContext(
        capture_key="rollout-1",
        model_call_id=call_id,
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        parent_resolution_mode="source_history",
        agent_instance=identity.agent_instance,
        model_ref=identity.model_ref,
    )
    token = set_token_sink(context)
    handler = VLLMWorkerCaptureHandler()
    try:
        await resolve_parent(items)
        admission = context.capture_admission
        await register_call_intent()
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
        return admission
    finally:
        reset_token_sink(token)


QUESTION = {"role": "user", "content": "question"}
SERVED = {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
FOLLOW_UP = {"role": "user", "content": "continue"}


async def test_source_history_proposes_only_a_call_of_the_request_s_own_lineage_scope(ledger):
    await _serve(ledger, "a1", [QUESTION], ASSISTANT)

    judge = await _serve(ledger, "j1", [QUESTION, SERVED, FOLLOW_UP], JUDGE)
    assistant = await _serve(ledger, "a2", [QUESTION, SERVED, FOLLOW_UP], ASSISTANT)

    assert judge.mode == "text" and judge.parent_call_id is None
    # The judge's lineage scope has no call yet, whatever the assistant recorded.
    assert judge.root_reason == "first_call"
    assert assistant.mode == "candidate" and assistant.parent_call_id == "a1"


async def test_source_history_roots_a_continuation_served_by_another_model_server(ledger):
    await _serve(ledger, "a1", [QUESTION], ASSISTANT)

    moved = await _serve(
        ledger, "b1", [QUESTION, SERVED, FOLLOW_UP], CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_B)
    )

    assert moved.mode == "text" and moved.parent_call_id is None
    assert moved.root_reason == "model_server_changed"
    manifest = RolloutManifest.model_validate(await ledger.manifest("rollout-1"))
    assert [record.root_reason for record in manifest.records] == ["first_call", "model_server_changed"]


# --- terminal attribution per agent instance ---


async def test_a_declared_terminal_is_attributed_to_the_primary_agent_instance_only(tmp_path):
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()])

    built = await trajectories_from_source(ROLLOUT, store, declared_response_id="resp_a1")

    judge = built[AGENT_INSTANCE_BUILDS_KEY]["judge"]
    assert built["mask_sample"] is False and _generation_tokens(built) == [[10, 11]]
    assert judge["mask_sample"] is False and _generation_tokens(judge) == [[9]]


def test_the_scored_response_is_attributed_to_the_primary_agent_instance_only(tmp_path):
    # The judge's output repeats the assistant's answer, so the scored response's text matches both.
    echo = _entry("j1", JUDGE, [USER, ANSWER, JUDGE_PROMPT], "the answer is 4", prompt=[7, 8], generation=[9])
    _store_with(tmp_path, [_assistant_call(), echo])
    scored = {"id": "resp_a1", "output": [{"type": "message", "role": "assistant", "content": "the answer is 4"}]}

    with_scored = trajectories_for_rollout(ROLLOUT, [tmp_path], verified_response=scored)
    without = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert with_scored["metrics"]["terminal_attribution"]["chain"] == "delivered"
    judge, judge_without = (build[AGENT_INSTANCE_BUILDS_KEY]["judge"] for build in (with_scored, without))
    assert judge["metrics"].get("terminal_attribution") == judge_without["metrics"].get("terminal_attribution")


def test_an_agent_instance_on_two_model_servers_delivers_the_attributed_server_s_chain(tmp_path):
    moved = CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_B)
    other = _entry("b1", moved, [USER], "a different answer", prompt=[4, 5], generation=[20, 21])
    _store_with(tmp_path, [_assistant_call(), other])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path], explicit_terminal_call_id="b1")

    assert built["metrics"]["model_servers"] == 2
    assert built["mask_sample"] is False
    assert _generation_tokens(built) == [[20, 21]]


def test_a_response_id_picks_the_model_server_when_both_served_the_scored_text(tmp_path):
    # Each server's call matches the scored text. Over all the agent instance's calls the text is ambiguous,
    # so the response id alone names the terminal call.
    moved = CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_B)
    other = _entry("b1", moved, [USER], "the answer is 4", prompt=[4, 5], generation=[20, 21])
    _store_with(tmp_path, [_assistant_call(), other])
    scored = {"id": "resp_b1", "output": [{"type": "message", "role": "assistant", "content": "the answer is 4"}]}

    built = trajectories_for_rollout(ROLLOUT, [tmp_path], verified_response=scored)

    assert built["metrics"]["terminal_attribution"]["call_id"] == "b1"
    assert built["mask_sample"] is False and _generation_tokens(built) == [[20, 21]]
    assert built["unresolved_retries"] == []


def test_witnesses_naming_calls_on_two_model_servers_deliver_neither(tmp_path):
    moved = CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_B)
    other = _entry("b1", moved, [USER], "a different answer", prompt=[4, 5], generation=[20, 21])
    _store_with(tmp_path, [_assistant_call(), other])
    scored = {"id": "resp_b1", "output": [{"type": "message", "role": "assistant", "content": "a different answer"}]}

    built = trajectories_for_rollout(ROLLOUT, [tmp_path], verified_response=scored, explicit_terminal_call_id="a1")

    assert built["mask_sample"] is True
    assert built["metrics"]["terminal_attribution"]["call_id"] is None


def test_identical_roots_on_two_model_servers_are_not_grouped_as_retries(tmp_path):
    moved = CallIdentity(agent_instance="assistant", model_ref=MODEL_SERVER_B)
    twin = _entry("a2", moved, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])
    _store_with(tmp_path, [_assistant_call(), twin])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert built["unresolved_retries"] == []
    assert built["metrics"]["model_servers"] == 2
    # Neither server's chain was attributed, so the agent instance's delivered chain is unknown.
    assert built["mask_sample"] is True


def test_identical_first_prompts_of_two_agent_instances_are_two_roots_not_retries(tmp_path):
    judge_call = _entry("j1", JUDGE, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])
    _store_with(tmp_path, [_assistant_call(), judge_call])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    judge = built[AGENT_INSTANCE_BUILDS_KEY]["judge"]
    assert built["mask_sample"] is False and built["metrics"]["roots"] == 1 and built["unresolved_retries"] == []
    assert judge["mask_sample"] is False and judge["unresolved_retries"] == []
    assert _generation_tokens(built) == [[10, 11]]
    assert set(built[AGENT_INSTANCE_BUILDS_KEY]) == {"assistant", "judge"}


def test_each_agent_instance_gets_its_own_terminal_chain(tmp_path):
    second = _entry(
        "a2",
        ASSISTANT,
        [USER, ANSWER, {"role": "user", "content": "and?"}],
        "done",
        prompt=[1, 2, 3, 10, 11, 4],
        generation=[12],
        parent="a1",
    )
    judge_call = _entry("j1", JUDGE, REPLAYED, "4/5", prompt=[1, 2, 3, 10, 11, 5], generation=[9])
    _store_with(tmp_path, [_assistant_call(), second, judge_call])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    judge = built[AGENT_INSTANCE_BUILDS_KEY]["judge"]
    assert _generation_tokens(built) == [[10, 11], [12]] and _generation_tokens(judge) == [[9]]
    assert built["mask_sample"] is False and judge["mask_sample"] is False


async def test_a_declared_terminal_naming_another_agent_instance_s_call_is_masked(tmp_path):
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()])

    built = await trajectories_from_source(ROLLOUT, store, declared_response_id="resp_j1")

    assert built["mask_sample"] is True
    assert "declared_terminal_not_captured" in built["metrics"]["terminal_attribution"]["reason"]


# --- the primary agent instance ---


def test_the_assistant_is_the_primary_before_the_unnamed_agent():
    assert primary_agent_instance([None, "assistant"]) == "assistant"
    assert primary_agent_instance(["judge", None]) is None
    assert primary_agent_instance([None]) is None
    assert primary_agent_instance(["judge"]) == "judge"
    with pytest.raises(LookupError):
        primary_agent_instance(["user", "judge"])


def test_an_unnamed_caller_beside_the_assistant_does_not_take_the_scored_response(tmp_path):
    # The unnamed call comes first, so only the rule, not the order of capture, makes the assistant primary.
    helper = _entry("h1", UNNAMED, [USER], "a helper's note", prompt=[4, 5], generation=[20, 21])
    _store_with(tmp_path, [helper, _assistant_call()])
    scored = {"id": "resp_a1", "output": [{"type": "message", "role": "assistant", "content": "the answer is 4"}]}

    built = trajectories_for_rollout(ROLLOUT, [tmp_path], verified_response=scored)

    assert built["mask_sample"] is False and _generation_tokens(built) == [[10, 11]]
    assert set(built[AGENT_INSTANCE_BUILDS_KEY]) == {"assistant"}


def test_named_agent_instances_without_an_assistant_have_no_primary(tmp_path):
    user_call = _entry("u1", USER_SIM, [USER], "ok", prompt=[1, 2], generation=[3])
    _store_with(tmp_path, [user_call, _judge_call()], intents=[("u1", USER_SIM), ("j1", JUDGE)])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert built["mask_sample"] is True and built["rebuilt_response"] is None
    assert built["error"] == "no primary agent instance captured"
    assert "capture_incomplete" not in built["metrics"]
    assert {name: build["mask_sample"] for name, build in built[AGENT_INSTANCE_BUILDS_KEY].items()} == {
        "user": False,
        "judge": False,
    }


def test_an_episode_wide_mark_is_reported_when_no_agent_instance_is_primary(tmp_path):
    user_call = _entry("u1", USER_SIM, [USER], "ok", prompt=[1, 2], generation=[3])
    _store_with(tmp_path, [user_call, _judge_call()], marks=[("unknown", None)])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert built["mask_sample"] is True and built["metrics"]["capture_incomplete"] is True


# --- outcomes ---


@pytest.mark.parametrize("mask_incomplete_when_attributed", [True, False])
def test_an_agent_instance_s_mark_masks_it_and_nobody_else(tmp_path, mask_incomplete_when_attributed):
    # The marked call has no intent, so only the mark itself carries the agent instance.
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()], marks=[("a9", ASSISTANT)])

    built = trajectories_for_rollout(
        ROLLOUT, [tmp_path], mask_incomplete_when_attributed=mask_incomplete_when_attributed
    )
    snapshot = store.freeze_now(ROLLOUT)

    judge = built[AGENT_INSTANCE_BUILDS_KEY]["judge"]
    assert built["metrics"]["capture_incomplete"] is True and built["mask_sample"] is True
    assert "capture_incomplete" not in judge["metrics"] and judge["mask_sample"] is False
    assert snapshot.incomplete_agent_instances == frozenset({"assistant"}) and not snapshot.incomplete_episode


def test_a_mark_naming_no_model_call_belongs_to_the_agent_instance_that_made_the_request(tmp_path):
    # The model server writes this mark for an unclassified request: no call id, the requester's identity.
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()], marks=[("", JUDGE)])

    snapshot = store.freeze_now(ROLLOUT)
    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert snapshot.incomplete_episode is False and snapshot.incomplete_agent_instances == frozenset({"judge"})
    assert built["mask_sample"] is False and "capture_incomplete" not in built["metrics"]
    assert built[AGENT_INSTANCE_BUILDS_KEY]["judge"]["mask_sample"] is True


def test_a_mark_naming_no_model_call_and_no_agent_instance_is_episode_wide_in_a_named_episode(tmp_path):
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()], marks=[("", UNNAMED)])

    snapshot = store.freeze_now(ROLLOUT)

    assert snapshot.incomplete_episode is True and snapshot.incomplete_agent_instances == frozenset()


@pytest.mark.parametrize("named_by", ["entry", "intent"])
def test_a_mark_without_an_agent_instance_is_episode_wide_once_the_episode_names_one(tmp_path, named_by):
    # The mark names its model server only. Either a named entry or a named intent makes the episode named.
    if named_by == "entry":
        store = _store_with(tmp_path, [_assistant_call()], marks=[("x1", UNNAMED)])
    else:
        store = _store_with(tmp_path, [], intents=[("a1", ASSISTANT)], marks=[("x1", UNNAMED)])

    snapshot = store.freeze_now(ROLLOUT)

    assert snapshot.incomplete_episode is True
    assert None not in snapshot.incomplete_agent_instances


def test_an_unnamed_episode_s_mark_belongs_to_the_unnamed_agent(tmp_path):
    call = _entry("c1", UNNAMED, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])
    store = _store_with(tmp_path, [call], intents=[("c1", UNNAMED)], marks=[("x1", UNNAMED)])

    snapshot = store.freeze_now(ROLLOUT)

    assert snapshot.incomplete is True
    assert snapshot.incomplete_agent_instances == frozenset({None}) and snapshot.incomplete_episode is False


@pytest.mark.parametrize("mask_incomplete_when_attributed", [True, False])
def test_a_capture_with_only_a_lost_call_is_reported_incomplete(tmp_path, mask_incomplete_when_attributed):
    _store_with(tmp_path, [], intents=[("c1", UNNAMED)])

    built = trajectories_for_rollout(
        ROLLOUT, [tmp_path], mask_incomplete_when_attributed=mask_incomplete_when_attributed
    )

    assert built["mask_sample"] is True and built["rebuilt_response"] is None
    assert built["metrics"]["capture_incomplete"] is True


def test_an_attributed_incomplete_capture_trains_when_configured(tmp_path):
    call = _entry("c1", UNNAMED, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])
    _store_with(tmp_path, [call], marks=[("x1", UNNAMED)])

    kept = trajectories_for_rollout(
        ROLLOUT, [tmp_path], explicit_terminal_call_id="c1", mask_incomplete_when_attributed=False
    )
    masked = trajectories_for_rollout(
        ROLLOUT, [tmp_path], explicit_terminal_call_id="c1", mask_incomplete_when_attributed=True
    )

    assert kept["metrics"]["capture_incomplete"] is True and kept["mask_sample"] is False
    assert masked["metrics"]["capture_incomplete"] is True and masked["mask_sample"] is True


@pytest.mark.parametrize("mask_incomplete_when_attributed", [True, False])
def test_an_assistant_s_own_lost_call_masks_it(tmp_path, mask_incomplete_when_attributed):
    _store_with(tmp_path, [_assistant_call()], intents=[("a1", ASSISTANT), ("a2", ASSISTANT)])

    built = trajectories_for_rollout(
        ROLLOUT, [tmp_path], mask_incomplete_when_attributed=mask_incomplete_when_attributed
    )

    # Without a scored response nothing places the lost call off the delivered chain.
    assert built["metrics"]["capture_incomplete"] is True and built["mask_sample"] is True


def test_a_lost_call_inherits_its_intent_s_agent_instance(tmp_path):
    store = _store_with(tmp_path, [_assistant_call()], intents=[("j1", JUDGE)], marks=[("j1", None)])

    snapshot = store.freeze_now(ROLLOUT)

    assert snapshot.incomplete is True and snapshot.incomplete_agent_instances == frozenset({"judge"})
    assert not snapshot.incomplete_for("assistant") and snapshot.incomplete_for("judge")


def test_a_single_agent_snapshot_keeps_its_flat_result(tmp_path):
    call = _entry("c1", UNNAMED, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])
    _store_with(tmp_path, [call], intents=[("c1", UNNAMED), ("c2", UNNAMED)])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert AGENT_INSTANCE_BUILDS_KEY not in built
    assert built["mask_sample"] is True and built["metrics"]["capture_incomplete"] is True


def test_an_episode_wide_mark_survives_a_crash_before_the_state_is_written(tmp_path, monkeypatch):
    # The mark line is written first. A state that says incomplete with no line would, beside the judge's
    # mark, be read as the judge's alone.
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()], marks=[("j-lost", JUDGE)])

    def crash(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(store, "_write_state", crash)
    with pytest.raises(OSError):
        store._mark_incomplete(ROLLOUT)

    snapshot = TokenCaptureStore(tmp_path).freeze_now(ROLLOUT)

    assert snapshot.incomplete_episode is True and snapshot.incomplete_for("assistant")


# --- an agent instance that lost every call ---


@pytest.mark.parametrize("mask_incomplete_when_attributed", [True, False])
def test_a_primary_agent_instance_whose_every_call_was_lost_is_not_replaced_by_another(
    tmp_path, mask_incomplete_when_attributed
):
    _store_with(tmp_path, [_judge_call()], intents=[("a1", ASSISTANT), ("j1", JUDGE)])

    built = trajectories_for_rollout(
        ROLLOUT, [tmp_path], mask_incomplete_when_attributed=mask_incomplete_when_attributed
    )

    builds = built[AGENT_INSTANCE_BUILDS_KEY]
    assert built["mask_sample"] is True and built["rebuilt_response"] is None
    assert built["metrics"]["capture_incomplete"] is True
    assert builds["assistant"]["mask_sample"] is True and builds["assistant"]["rebuilt_response"] is None
    assert builds["judge"]["mask_sample"] is False and _generation_tokens(builds["judge"]) == [[9]]
    assert "capture_incomplete" not in builds["judge"]["metrics"]


def test_a_lost_agent_instance_leaves_a_lone_captured_one_without_a_primary(tmp_path):
    # Had the judge's call been captured, neither would be the primary. Losing it must not make the user one.
    user_call = _entry("u1", USER_SIM, [USER], "ok", prompt=[1, 2], generation=[3])
    _store_with(tmp_path, [user_call], intents=[("j1", JUDGE)])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    builds = built[AGENT_INSTANCE_BUILDS_KEY]
    assert built["mask_sample"] is True and built["rebuilt_response"] is None
    assert built["error"] == "no primary agent instance captured"
    assert builds["user"]["mask_sample"] is False and _generation_tokens(builds["user"]) == [[3]]
    assert builds["judge"]["mask_sample"] is True and builds["judge"]["rebuilt_response"] is None


def test_lost_agent_instances_follow_the_captured_ones_in_name_order(tmp_path):
    lost = [
        (f"{name}-1", CallIdentity(agent_instance=name, model_ref=MODEL_SERVER_A)) for name in ("user", "debater", "b")
    ]
    _store_with(tmp_path, [_judge_call()], intents=lost)

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert list(built[AGENT_INSTANCE_BUILDS_KEY]) == ["judge", "b", "debater", "user"]


def test_an_agent_instance_that_lost_one_of_its_calls_keeps_its_captured_chain(tmp_path):
    _store_with(tmp_path, [_assistant_call(), _judge_call()], intents=[("a1", ASSISTANT), ("a2", ASSISTANT)])

    built = trajectories_for_rollout(
        ROLLOUT, [tmp_path], explicit_terminal_call_id="a1", mask_incomplete_when_attributed=False
    )

    assert built["metrics"]["capture_incomplete"] is True and built["mask_sample"] is False
    assert _generation_tokens(built) == [[10, 11]]


# --- an episode-wide failure beside an agent instance's mark ---


def _conflicting_write(store: TokenCaptureStore) -> None:
    with pytest.raises(ValueError, match="reused with a different payload"):
        store.append(_assistant_call().model_copy(update={"generation_token_ids": [12, 13]}))


def _judge_mark(store: TokenCaptureStore) -> None:
    store._mark_incomplete(ROLLOUT, "j-lost", JUDGE)


def _mark_without_a_call_or_an_identity(store: TokenCaptureStore) -> None:
    store._mark_incomplete(ROLLOUT)


@pytest.mark.parametrize("episode_wide", [_conflicting_write, _mark_without_a_call_or_an_identity])
@pytest.mark.parametrize("judge_mark_first", [True, False])
def test_an_agent_instance_s_mark_does_not_hide_an_episode_wide_failure(tmp_path, episode_wide, judge_mark_first):
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()])
    for step in (_judge_mark, episode_wide) if judge_mark_first else (episode_wide, _judge_mark):
        step(store)

    snapshot = store.freeze_now(ROLLOUT)
    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert snapshot.incomplete_episode is True
    assert snapshot.incomplete_for("assistant") and snapshot.incomplete_for("judge")
    assert built["mask_sample"] is True and built["metrics"]["capture_incomplete"] is True


def test_a_conflicting_write_stays_episode_wide_through_later_writes(tmp_path):
    store = _store_with(tmp_path, [_assistant_call()])
    _conflicting_write(store)
    store._begin_call(ROLLOUT, "j1", JUDGE)
    store.append(_judge_call())
    _judge_mark(store)

    snapshot = TokenCaptureStore(tmp_path).freeze_now(ROLLOUT)

    assert snapshot.incomplete_episode is True and snapshot.incomplete_for("assistant")


def test_a_state_marked_incomplete_before_conflicts_were_recorded_is_episode_wide(tmp_path):
    # A state file written before ``incomplete_episode`` existed carries only the aggregate flag.
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()])
    _conflicting_write(store)
    state_path = store.state_path_for(ROLLOUT)
    state = json.loads(state_path.read_text())
    assert state.pop("incomplete_episode") is True and state["incomplete"] is True
    state_path.write_text(json.dumps(state))

    snapshot = store.freeze_now(ROLLOUT)

    assert snapshot.incomplete_episode is True
    assert snapshot.incomplete_for("assistant") and snapshot.incomplete_for("judge")


# --- failure identity in the manifest ---


async def test_a_failure_naming_only_its_model_server_inherits_the_intent_s_agent_instance(ledger):
    await ledger.begin_call("r1", "j1", identity=JUDGE)
    await ledger.record_failure("r1", "j1", "worker_capture_failed", identity=UNNAMED)

    [failure] = RolloutManifest.model_validate(await ledger.manifest("r1")).failures

    assert failure.agent_instance == "judge" and failure.model_ref == MODEL_SERVER_A
