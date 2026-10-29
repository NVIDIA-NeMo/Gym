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
"""Lineage, chains and outcomes are computed per agent instance and model server."""

import pytest

from nemo_gym.token_id_capture import trajectories_for_rollout
from nemo_gym.token_id_capture.config import check_one_capture_directory
from nemo_gym.token_id_capture.consumer import AGENT_INSTANCE_BUILDS_KEY
from nemo_gym.token_id_capture.lineage import (
    MODEL_SERVER_CHANGED_REASON,
    FileLineageStore,
    InMemoryLineageStore,
    stamp_continuation,
)
from nemo_gym.token_id_capture.records import (
    CallIdentity,
    ModelRef,
    ParentResolutionStatus,
    TokenEntry,
    stamp_lineage,
)
from nemo_gym.token_id_capture.staging.records import CallRecord
from nemo_gym.token_id_capture.staging.terminal import select_terminal_calls
from nemo_gym.token_id_capture.store import TokenCaptureStore


ROLLOUT = "t0-r0"
POLICY = ModelRef(name="policy")
OTHER_SERVER = ModelRef(name="policy-b")
ASSISTANT = CallIdentity(agent_instance="assistant", model_ref=POLICY)
JUDGE = CallIdentity(agent_instance="judge", model_ref=POLICY)

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
        rollout_id=ROLLOUT,
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


ASSISTANT_CALL = _entry("a1", ASSISTANT, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])


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


async def test_a_judge_prompt_extending_the_assistant_transcript_gets_no_parent(lineage):
    store, commit = lineage
    await commit(ASSISTANT_CALL)
    judge_request = [USER, ANSWER, JUDGE_PROMPT]

    judge = await store.resolve(ROLLOUT, judge_request, identity=JUDGE)
    assistant = await store.resolve(ROLLOUT, judge_request, identity=ASSISTANT)

    assert judge.match is None
    assert assistant.status == ParentResolutionStatus.RESOLVED and assistant.match.model_call_id == "a1"


async def test_a_continuation_on_another_model_server_is_a_root(lineage):
    store, commit = lineage
    await commit(ASSISTANT_CALL)
    request = [USER, ANSWER, {"role": "user", "content": "continue"}]

    moved = await store.resolve(
        ROLLOUT, request, identity=CallIdentity(agent_instance="assistant", model_ref=OTHER_SERVER)
    )

    assert moved.status == ParentResolutionStatus.ROOT
    assert moved.reason == MODEL_SERVER_CHANGED_REASON


async def test_an_unnamed_request_does_not_see_named_calls(lineage):
    store, commit = lineage
    await commit(ASSISTANT_CALL)

    resolution = await store.resolve(ROLLOUT, [USER, ANSWER, JUDGE_PROMPT], identity=CallIdentity(model_ref=POLICY))

    assert resolution.match is None


async def test_retire_leaves_no_group_in_the_lineage_caches(lineage):
    store, commit = lineage
    await commit(ASSISTANT_CALL)
    await commit(_entry("j1", JUDGE, [USER, ANSWER, JUDGE_PROMPT], "4/5", prompt=[7, 8], generation=[9]))
    await store.resolve(ROLLOUT, [USER, ANSWER, JUDGE_PROMPT], identity=ASSISTANT)
    await store.resolve(ROLLOUT, [USER, ANSWER, JUDGE_PROMPT], identity=JUDGE)

    await store.retire([ROLLOUT])

    if isinstance(store, InMemoryLineageStore):
        assert len(store.index) == 0
    else:
        assert ROLLOUT not in store._cache and ROLLOUT not in store._materialized


# --- chains and outcomes from a frozen local snapshot ---


def _freeze_and_build(tmp_path, entries, *, intents=(), marks=(), **kwargs) -> dict:
    store = TokenCaptureStore(tmp_path)
    for model_call_id, identity in intents:
        store._begin_call(ROLLOUT, model_call_id, identity)
    for entry in entries:
        store.append(entry)
    for model_call_id, identity in marks:
        store._mark_incomplete(ROLLOUT, model_call_id, identity)
    return trajectories_for_rollout(ROLLOUT, [tmp_path], **kwargs)


def test_identical_first_prompts_of_two_agent_instances_are_two_roots_not_retries(tmp_path):
    judge_call = _entry("j1", JUDGE, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])

    built = _freeze_and_build(tmp_path, [ASSISTANT_CALL, judge_call])

    assert built["mask_sample"] is False
    assert built["metrics"]["roots"] == 1 and built["unresolved_retries"] == []
    judge = built[AGENT_INSTANCE_BUILDS_KEY]["judge"]
    assert judge["mask_sample"] is False and judge["unresolved_retries"] == []
    assert built["rebuilt_response"]["output"][0]["generation_token_ids"] == [10, 11]
    assert set(built[AGENT_INSTANCE_BUILDS_KEY]) == {"assistant", "judge"}


def test_identical_roots_on_two_model_servers_are_not_grouped_as_retries(tmp_path):
    moved = CallIdentity(agent_instance="assistant", model_ref=OTHER_SERVER)
    twin = _entry("a2", moved, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])

    built = _freeze_and_build(tmp_path, [ASSISTANT_CALL, twin])

    assert built["unresolved_retries"] == []
    assert built["metrics"]["model_servers"] == 2
    # Neither server's chain was attributed, so the agent instance's delivered chain is unknown.
    assert built["mask_sample"] is True


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
    judge_call = _entry("j1", JUDGE, [USER, ANSWER, JUDGE_PROMPT], "4/5", prompt=[1, 2, 3, 10, 11, 5], generation=[9])

    built = _freeze_and_build(tmp_path, [ASSISTANT_CALL, second, judge_call])

    assistant_tokens = [item["generation_token_ids"] for item in built["rebuilt_response"]["output"]]
    judge = built[AGENT_INSTANCE_BUILDS_KEY]["judge"]
    assert assistant_tokens == [[10, 11], [12]]
    assert [item["generation_token_ids"] for item in judge["rebuilt_response"]["output"]] == [[9]]
    assert built["mask_sample"] is False and judge["mask_sample"] is False


async def test_a_declared_terminal_naming_another_agent_instance_s_call_is_masked(tmp_path):
    from nemo_gym.token_id_capture import trajectories_from_source

    judge_call = _entry("j1", JUDGE, [USER, ANSWER, JUDGE_PROMPT], "4/5", prompt=[7, 8], generation=[9])
    store = TokenCaptureStore(tmp_path)
    for entry in (ASSISTANT_CALL, judge_call):
        store.append(entry)

    built = await trajectories_from_source(ROLLOUT, store, declared_response_id="resp_j1")

    assert built["mask_sample"] is True
    assert "declared_terminal_not_captured" in built["metrics"]["terminal_attribution"]["reason"]


@pytest.mark.parametrize("mask_incomplete_when_attributed", [True, False])
def test_a_judge_s_lost_call_never_masks_the_assistant(tmp_path, mask_incomplete_when_attributed):
    built = _freeze_and_build(
        tmp_path,
        [ASSISTANT_CALL],
        intents=[("a1", ASSISTANT), ("j1", JUDGE)],
        marks=[("j2", JUDGE)],
        mask_incomplete_when_attributed=mask_incomplete_when_attributed,
    )

    assert built["mask_sample"] is False
    assert "capture_incomplete" not in built["metrics"]


@pytest.mark.parametrize("mask_incomplete_when_attributed", [True, False])
def test_an_assistant_s_own_lost_call_masks_it(tmp_path, mask_incomplete_when_attributed):
    built = _freeze_and_build(
        tmp_path,
        [ASSISTANT_CALL],
        intents=[("a1", ASSISTANT), ("a2", ASSISTANT)],
        mask_incomplete_when_attributed=mask_incomplete_when_attributed,
    )

    assert built["metrics"]["capture_incomplete"] is True
    # Without a scored response nothing places the lost call off the delivered chain.
    assert built["mask_sample"] is True


@pytest.mark.parametrize("mark", [("", None), ("unknown", None)])
def test_an_episode_wide_mark_masks_every_agent_instance(tmp_path, mark):
    judge_call = _entry("j1", JUDGE, [USER, ANSWER, JUDGE_PROMPT], "4/5", prompt=[7, 8], generation=[9])

    built = _freeze_and_build(tmp_path, [ASSISTANT_CALL, judge_call], marks=[mark])

    assert built["mask_sample"] is True
    assert built[AGENT_INSTANCE_BUILDS_KEY]["judge"]["mask_sample"] is True


def test_a_lost_call_inherits_its_intent_s_agent_instance(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store._begin_call(ROLLOUT, "j1", JUDGE)
    store.append(ASSISTANT_CALL)
    store._mark_incomplete(ROLLOUT, "j1")

    snapshot = store.freeze_now(ROLLOUT)

    assert snapshot.incomplete is True
    assert snapshot.incomplete_agent_instances == frozenset({"judge"})
    assert not snapshot.incomplete_for("assistant") and snapshot.incomplete_for("judge")


def test_a_single_agent_snapshot_keeps_its_flat_result(tmp_path):
    unnamed = CallIdentity(model_ref=POLICY)
    call = _entry("c1", unnamed, [USER], "the answer is 4", prompt=[1, 2, 3], generation=[10, 11])

    built = _freeze_and_build(tmp_path, [call], intents=[("c1", unnamed), ("c2", unnamed)])

    assert AGENT_INSTANCE_BUILDS_KEY not in built
    assert built["mask_sample"] is True and built["metrics"]["capture_incomplete"] is True


# --- manifest terminal selection per group ---


def _call_record(model_call_id: str, identity: CallIdentity, parent: str | None = None) -> CallRecord:
    return CallRecord(
        model_call_id=model_call_id,
        parent_call_id=parent,
        prev_len=4 if parent else 0,
        delta_len=4,
        cum_len=8 if parent else 4,
        weight_version=0,
        digest="a" * 64,
        extras_digest="b" * 64,
        staging_key=f"{ROLLOUT}/{model_call_id}",
        mode="token_in" if parent else "text",
        chain_hash="c" * 64,
        cumulative_hash="d" * 64,
        response_id=f"resp_{model_call_id}",
        agent_instance=identity.agent_instance,
        agent_kind=identity.agent_kind,
        model_ref=identity.model_ref,
    )


def test_manifest_terminal_selection_runs_once_per_group():
    records = [
        _call_record("a1", ASSISTANT),
        _call_record("a2", ASSISTANT, parent="a1"),
        _call_record("j1", JUDGE),
    ]

    selections = select_terminal_calls(records)

    assert selections[("assistant", "policy")].terminal_model_call_id == "a2"
    assert selections[("judge", "policy")].terminal_model_call_id == "j1"


# --- one capture directory per run ---


def _run(**model_servers) -> dict:
    config = {"token_id_capture": {"enabled": True, "dir": "/capture"}}
    for name, overrides in model_servers.items():
        config[name] = {"responses_api_models": {"vllm_model": {"host": "h", "port": 1, **overrides}}}
    return config


def test_model_servers_sharing_the_capture_directory_are_accepted():
    check_one_capture_directory(_run(policy={}, judge={"token_id_capture": {"dir": "/capture"}}))


def test_model_servers_with_different_capture_directories_are_refused():
    with pytest.raises(ValueError, match="share one capture directory"):
        check_one_capture_directory(_run(policy={}, judge={"token_id_capture": {"dir": "/elsewhere"}}))


def test_a_custom_sink_owns_its_namespace():
    config = _run(policy={}, judge={"token_id_capture": {"dir": "/elsewhere"}})
    config["token_id_capture"].update(sink="pkg.module:Sink", lineage_store="pkg.module:Store")
    check_one_capture_directory(config)
