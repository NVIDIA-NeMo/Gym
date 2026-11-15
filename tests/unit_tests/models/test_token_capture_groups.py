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
"""Group scoping of parent search, terminal attribution, outcomes, manifest filters and the startup check.

A group is the calls of one agent instance on one model server (``nemo_gym.models.token_capture.groups``).
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.client.server_client import ServerClient
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.models.token_capture.config import check_one_capture_directory
from nemo_gym.models.token_capture.consumer import (
    AGENT_INSTANCE_BUILDS_KEY,
    trajectories_for_rollout,
    trajectories_from_source,
)
from nemo_gym.models.token_capture.control_routes import (
    CONTROL_ROUTE_PREFIX,
    RolloutControlClient,
    install_rollout_control_routes,
)
from nemo_gym.models.token_capture.delivery import finalize_rollout_token_capture
from nemo_gym.models.token_capture.external_capture import VLLMWorkerCaptureHandler
from nemo_gym.models.token_capture.lineage import (
    FileLineageStore,
    InMemoryLineageStore,
    stamp_continuation,
)
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
POLICY = ModelRef(name="policy")
OTHER_SERVER = ModelRef(name="policy-b")
UNNAMED = CallIdentity(model_ref=POLICY)
ASSISTANT = CallIdentity(agent_instance="assistant", model_ref=POLICY)
JUDGE = CallIdentity(agent_instance="judge", model_ref=POLICY)
USER_SIM = CallIdentity(agent_instance="user", model_ref=POLICY)

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
) -> TokenEntry:
    entry = TokenEntry(
        capture_key=ROLLOUT,
        model_call_id=model_call_id,
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        generation_log_probs=[-0.1] * len(generation),
        output_items=[{"type": "message", "role": "assistant", "content": output_text}],
        response_id=f"resp_{model_call_id}",
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


async def test_another_agent_instance_s_call_on_another_model_server_is_not_a_model_server_change(lineage):
    store, commit = lineage
    await commit(_assistant_call())

    resolution = await store.resolve(
        ROLLOUT,
        [USER, ANSWER, JUDGE_PROMPT],
        identity=CallIdentity(agent_instance="judge", model_ref=OTHER_SERVER),
    )

    assert resolution.status == ParentResolutionStatus.UNRESOLVED
    assert resolution.reason == "no_match"


async def test_a_request_no_model_server_served_is_not_a_model_server_change(lineage):
    store, commit = lineage
    await commit(_assistant_call())
    never_served = [USER, {"role": "assistant", "content": "nobody served this"}, JUDGE_PROMPT]

    resolution = await store.resolve(
        ROLLOUT, never_served, identity=CallIdentity(agent_instance="assistant", model_ref=OTHER_SERVER)
    )

    assert resolution.status == ParentResolutionStatus.UNRESOLVED
    assert resolution.reason == "no_match"


async def test_a_judge_request_replaying_the_assistant_s_turns_is_unresolved_and_masks_only_the_judge(
    lineage, tmp_path_factory
):
    # The judge's group holds no call that served the replayed assistant turn, and a request with model
    # history is never a root, so the judge's own sample is masked. The assistant's is untouched.
    store, commit = lineage
    await commit(_assistant_call())
    replayed = [USER, ANSWER, JUDGE_PROMPT]
    transcript_as_text = [{"role": "user", "content": "Grade this transcript: solve the task / the answer is 4"}]

    resolution = await store.resolve(ROLLOUT, replayed, identity=JUDGE)
    as_text = await store.resolve(ROLLOUT, transcript_as_text, identity=JUDGE)

    assert (resolution.status, resolution.reason) == (ParentResolutionStatus.UNRESOLVED, "no_match")
    assert as_text.status == ParentResolutionStatus.ROOT
    judge_call = _judge_call()
    stamp_lineage(judge_call, None, parent_resolution=resolution.status)
    directory = tmp_path_factory.mktemp("build")
    _store_with(directory, [_assistant_call(), judge_call])
    built = trajectories_for_rollout(ROLLOUT, [directory])
    assert built["mask_sample"] is False and _generation_tokens(built) == [[10, 11]]
    assert built[AGENT_INSTANCE_BUILDS_KEY]["judge"]["mask_sample"] is True


# --- source-history admission (worker custody) ---


@pytest.fixture(params=["memory", "file"])
def ledger(request, tmp_path):
    return InMemoryLineageStore() if request.param == "memory" else FileLineageStore(tmp_path)


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


async def test_source_history_proposes_only_a_call_of_the_request_s_own_group(ledger):
    await _serve(ledger, "a1", [QUESTION], ASSISTANT)

    judge = await _serve(ledger, "j1", [QUESTION, SERVED, FOLLOW_UP], JUDGE)
    assistant = await _serve(ledger, "a2", [QUESTION, SERVED, FOLLOW_UP], ASSISTANT)

    assert judge.mode == "text" and judge.parent_call_id is None
    # The judge's group has no call yet, whatever the assistant recorded.
    assert judge.root_reason == "first_call"
    assert assistant.mode == "candidate" and assistant.parent_call_id == "a1"


async def test_source_history_roots_a_continuation_served_by_another_model_server(ledger):
    await _serve(ledger, "a1", [QUESTION], ASSISTANT)

    moved = await _serve(
        ledger, "b1", [QUESTION, SERVED, FOLLOW_UP], CallIdentity(agent_instance="assistant", model_ref=OTHER_SERVER)
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
    moved = CallIdentity(agent_instance="assistant", model_ref=OTHER_SERVER)
    other = _entry("b1", moved, [USER], "a different answer", prompt=[4, 5], generation=[20, 21])
    _store_with(tmp_path, [_assistant_call(), other])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path], explicit_terminal_call_id="b1")

    assert built["metrics"]["model_servers"] == 2
    assert built["mask_sample"] is False
    assert _generation_tokens(built) == [[20, 21]]


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


def test_a_mark_naming_no_model_call_is_episode_wide_whoever_made_the_request(tmp_path):
    # The model server writes this mark for an unclassified request: no call id, the requester's identity.
    store = _store_with(tmp_path, [_assistant_call(), _judge_call()], marks=[("", JUDGE)])

    snapshot = store.freeze_now(ROLLOUT)
    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert snapshot.incomplete_episode is True and snapshot.incomplete_agent_instances == frozenset()
    assert built["mask_sample"] is True and built[AGENT_INSTANCE_BUILDS_KEY]["judge"]["mask_sample"] is True


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


# The assistant's response, scored by the verifier, in an episode whose capture holds no call of the assistant.
UNCAPTURED_ASSISTANT_RESPONSE = {
    "id": "resp_assistant",
    "output": [{"type": "message", "role": "assistant", "content": "the answer is 4"}],
}


@pytest.mark.parametrize("identity", [JUDGE, USER_SIM, CallIdentity(agent_instance="debater.1", model_ref=POLICY)])
def test_the_only_captured_agent_instance_is_not_the_primary_unless_it_is_the_assistant(tmp_path, identity):
    # ``training_agent_instances: [judge]`` tags the judge's URL only, so the ledger never sees the assistant.
    call = _entry("c1", identity, [USER, ANSWER, JUDGE_PROMPT], "4/5", prompt=[7, 8], generation=[9])
    _store_with(tmp_path, [call], intents=[("c1", identity)])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path], verified_response=UNCAPTURED_ASSISTANT_RESPONSE)

    assert built["mask_sample"] is True and built["rebuilt_response"] is None
    assert built["error"] == "no primary agent instance captured"
    own = built[AGENT_INSTANCE_BUILDS_KEY][identity.agent_instance]
    assert own["mask_sample"] is False and _generation_tokens(own) == [[9]]


def test_the_only_captured_agent_instance_is_the_primary_when_it_is_the_assistant(tmp_path):
    _store_with(tmp_path, [_assistant_call()], intents=[("a1", ASSISTANT)])

    built = trajectories_for_rollout(ROLLOUT, [tmp_path])

    assert built["mask_sample"] is False and _generation_tokens(built) == [[10, 11]]
    assert _generation_tokens(built[AGENT_INSTANCE_BUILDS_KEY]["assistant"]) == [[10, 11]]


async def test_finalization_keeps_the_scored_output_when_only_another_agent_instance_was_captured(tmp_path):
    store = _store_with(tmp_path, [_judge_call()], intents=[("j1", JUDGE)])
    scored_output = [dict(item) for item in UNCAPTURED_ASSISTANT_RESPONSE["output"]]
    result = {
        "_ng_rollout_id": ROLLOUT,
        "reward": 0.75,
        "response": {**UNCAPTURED_ASSISTANT_RESPONSE, "model": "m"},
    }

    built = await finalize_rollout_token_capture(result, store)

    # The judge's tokens never become the episode's sample: the scored output stays, masked.
    assert result["response"]["output"] == scored_output
    assert result["reward"] == 0.75 and result["mask_sample"] is True
    assert _generation_tokens(built[AGENT_INSTANCE_BUILDS_KEY]["judge"]) == [[9]]


def test_lost_agent_instances_follow_the_captured_ones_in_name_order(tmp_path):
    lost = [(f"{name}-1", CallIdentity(agent_instance=name, model_ref=POLICY)) for name in ("user", "debater", "b")]
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


# --- manifest filters ---


async def test_a_failure_naming_only_its_model_server_inherits_the_intent_s_agent_instance(ledger):
    await ledger.begin_call("r1", "j1", identity=JUDGE)
    await ledger.record_failure("r1", "j1", "worker_capture_failed", identity=UNNAMED)

    [failure] = RolloutManifest.model_validate(await ledger.manifest("r1")).failures
    by_assistant = RolloutManifest.model_validate(await ledger.manifest("r1", agent_instance="assistant"))

    assert failure.agent_instance == "judge" and failure.model_ref == POLICY
    assert by_assistant.failures == []


async def test_a_failure_without_an_agent_instance_is_episode_wide_in_a_named_episode(ledger):
    await ledger.begin_call("r1", "a1", identity=ASSISTANT)
    await ledger.record_failure("r1", "x1", "worker_capture_failed", identity=UNNAMED)

    by_judge = RolloutManifest.model_validate(await ledger.manifest("r1", agent_instance="judge"))

    assert [failure.model_call_id for failure in by_judge.failures] == ["x1"]
    assert by_judge.attempted_call_ids == []


async def test_filters_of_an_unnamed_episode_keep_only_matching_and_unidentified_failures(ledger):
    await ledger.begin_call("r1", "c1", identity=UNNAMED)
    await ledger.record_failure("r1", "c1", "worker_capture_failed", identity=UNNAMED)
    await ledger.record_failure("r1", "bare", "worker_capture_failed")
    await ledger.begin_call("r1", "pending-bare")

    elsewhere = RolloutManifest.model_validate(await ledger.manifest("r1", model_ref="elsewhere"))
    here = RolloutManifest.model_validate(await ledger.manifest("r1", model_ref="policy"))

    # Only failures are episode-wide: a pending call with no identity is not returned under a filter.
    assert [failure.model_call_id for failure in elsewhere.failures] == ["bare"]
    assert elsewhere.attempted_call_ids == [] and elsewhere.pending_call_ids == []
    assert [failure.model_call_id for failure in here.failures] == ["c1", "bare"]
    assert here.attempted_call_ids == ["c1"]


async def test_the_route_and_the_client_filter_a_manifest_by_agent_instance(tmp_path):
    ledger = FileLineageStore(tmp_path)
    await ledger.record_failure("r1", "a1", "worker_capture_failed", identity=ASSISTANT)
    await ledger.record_failure("r1", "j1", "worker_capture_failed", identity=JUDGE)
    app = FastAPI()
    install_rollout_control_routes(app, ledger, auth_token="secret")
    http = TestClient(app)
    seen = []

    class _Response:
        def __init__(self, response):
            self.status = response.status_code
            self._payload = response.json()

        async def json(self):
            return self._payload

        async def text(self):
            return str(self._payload)

    async def request(method, path, **kwargs):
        seen.append(kwargs.get("params"))
        return _Response(
            http.request(
                method,
                f"{CONTROL_ROUTE_PREFIX}{path}",
                params=kwargs.get("params"),
                headers={"Authorization": "Bearer secret"},
            )
        )

    control = RolloutControlClient("http://model", auth_token="secret", request_timeout_s=1.0)
    control._request = request

    manifest = await control.manifest("r1", agent_instance="judge")

    assert seen == [{"agent_instance": "judge"}]
    assert [failure.model_call_id for failure in manifest.failures] == ["j1"]


# --- one capture directory per run ---


def _run(**model_servers) -> dict:
    config = {"token_id_capture": {"enabled": True}, "model_call_capture_dir": "/capture"}
    for name, overrides in model_servers.items():
        config[name] = {"responses_api_models": {"vllm_model": {"host": "h", "port": 1, **overrides}}}
    return config


def test_a_model_server_with_its_own_model_call_capture_dir_is_refused():
    with pytest.raises(ValueError, match="share one capture directory"):
        check_one_capture_directory(_run(policy={}, judge={"model_call_capture_dir": "/elsewhere"}))


class _Model(SimpleResponsesAPIModel):
    config: BaseResponsesAPIModelConfig

    async def responses(self, body=None):
        raise NotImplementedError

    async def chat_completions(self, body=None):
        raise NotImplementedError


def test_a_model_server_refuses_to_start_when_capture_directories_differ(tmp_path):
    config = {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path / "shared"),
        "token_id_capture": {"enabled": True, "dir": str(tmp_path / "shared")},
        "policy": {"responses_api_models": {"vllm_model": {"host": "h", "port": 1}}},
        "judge": {
            "responses_api_models": {
                "vllm_model": {"host": "h", "port": 2, "token_id_capture": {"dir": str(tmp_path / "judge")}}
            }
        },
    }
    model = _Model(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="policy"),
        server_client=MagicMock(spec=ServerClient, global_config_dict=config),
    )

    with pytest.raises(ValueError, match="share one capture directory"):
        model.setup_webserver()
