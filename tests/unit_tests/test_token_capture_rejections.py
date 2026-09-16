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

from unittest.mock import AsyncMock

import orjson
import pytest

from nemo_gym.token_id_capture import (
    CaptureContext,
    TokenCaptureStore,
    TokenEntry,
    capture_tokens,
    record_call_rejection,
    register_call_intent,
    reset_token_sink,
    set_token_sink,
)
from nemo_gym.token_id_capture.consumer import trajectories_for_rollout, trajectories_from_source
from nemo_gym.token_id_capture.protocols import CallRejection
from nemo_gym.token_id_capture.records import ParentResolutionStatus


def entry(call_id: str) -> TokenEntry:
    return TokenEntry(
        rollout_id="episode",
        model_call_id=call_id,
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.5],
        parent_resolution=ParentResolutionStatus.ROOT,
    )


@pytest.mark.parametrize("consumer", ["local", "source"])
@pytest.mark.parametrize("outcome", ["rejected", "unknown", "rejected_and_lost"])
async def test_only_known_rejections_preserve_successful_calls(tmp_path, consumer, outcome):
    store = TokenCaptureStore(tmp_path)
    await store.begin_call("episode", "before")
    await store.put(entry("before"))
    await store.begin_call("episode", "overflow")
    if outcome != "unknown":
        await store.reject_call("episode", "overflow", reason="context_length_exceeded")
    if outcome == "rejected_and_lost":
        await store.begin_call("episode", "lost-response")
    await store.begin_call("episode", "after")
    # Independent full contexts remain usable after compaction/retokenization.
    await store.put(entry("after").model_copy(update={"parent_resolution": ParentResolutionStatus.UNRESOLVED}))

    # Reopen as a different consumer; no process-local acknowledgement is needed.
    reader = TokenCaptureStore(tmp_path)
    kwargs = dict(builder="independent_calls", explicit_terminal_call_id="after")
    if consumer == "local":
        built = trajectories_for_rollout("episode", [tmp_path], **kwargs)
    else:
        built = await trajectories_from_source("episode", reader, **kwargs)
    assert built["mask_sample"] is (outcome != "rejected")
    assert built["metrics"]["generated_tokens_delivered"] == 2
    assert built["metrics"]["delivered_fraction"] == 1.0
    assert built["metrics"]["unresolved_parent_calls"] == 1
    assert built["metrics"]["rejected_without_generation_calls"] == (outcome != "unknown")
    assert built["metrics"]["pre_generation_rejections_by_reason"] == (
        {} if outcome == "unknown" else {"context_length_exceeded": 1}
    )
    assert {record.model_call_id for record in reader.freeze_now("episode").entries} == {"before", "after"}


async def test_rejection_is_idempotent_durable_and_retired_with_snapshot(tmp_path):
    store = TokenCaptureStore(tmp_path)
    await store.begin_call("episode", "overflow")
    await store.reject_call("episode", "overflow", reason="context_length_exceeded")
    state = store.state_path_for("episode").read_bytes()
    await store.reject_call("episode", "overflow", reason="context_length_exceeded")
    assert store.state_path_for("episode").read_bytes() == state

    reader = TokenCaptureStore(tmp_path)
    snapshot = await reader.freeze("episode")
    assert snapshot.rejected_calls == (CallRejection("overflow", "context_length_exceeded"),)
    assert not snapshot.incomplete
    assert await reader.freeze("episode") == snapshot
    assert await reader.drop("episode", snapshot_id=snapshot.snapshot_id, version=snapshot.version)
    assert "rejected_calls" not in orjson.loads(reader.state_path_for("episode").read_bytes())
    with pytest.raises(RuntimeError, match="retired"):
        await reader.reject_call("episode", "overflow", reason="context_length_exceeded")


async def test_late_rejection_cannot_rewrite_a_frozen_incomplete_snapshot(tmp_path):
    store = TokenCaptureStore(tmp_path)
    await store.begin_call("episode", "overflow")
    snapshot = await store.freeze("episode")
    assert snapshot.incomplete
    with pytest.raises(RuntimeError, match="already frozen"):
        await store.reject_call("episode", "overflow", reason="context_length_exceeded")
    assert await store.freeze("episode") == snapshot


@pytest.mark.parametrize("order", ["entry_first", "rejection_first", "conflicting_reason"])
async def test_conflicting_terminal_outcomes_remain_incomplete(tmp_path, order):
    store = TokenCaptureStore(tmp_path)
    await store.begin_call("episode", "call")
    if order == "entry_first":
        await store.put(entry("call"))
        with pytest.raises(ValueError, match="conflicting terminal outcomes"):
            await store.reject_call("episode", "call", reason="context_length_exceeded")
    else:
        await store.reject_call("episode", "call", reason="context_length_exceeded")
        with pytest.raises(ValueError, match="rejected|conflicting"):
            if order == "rejection_first":
                await store.put(entry("call"))
            else:
                await store.reject_call("episode", "call", reason="different-rejection")
    assert (await store.freeze("episode")).incomplete


async def test_rejection_cannot_clear_explicit_capture_failure(tmp_path):
    store = TokenCaptureStore(tmp_path)
    await store.begin_call("episode", "call")
    await store.mark_incomplete("episode", "call")
    await store.reject_call("episode", "call", reason="context_length_exceeded")
    assert (await store.freeze("episode")).incomplete


async def test_rejection_requires_intent_and_does_not_allow_call_id_reuse(tmp_path):
    store = TokenCaptureStore(tmp_path)
    with pytest.raises(ValueError, match="no registered intent"):
        await store.reject_call("episode", "call", reason="context_length_exceeded")
    await store.begin_call("episode", "call")
    with pytest.raises(ValueError, match="requires a reason"):
        await store.reject_call("episode", "call", reason="")
    await store.reject_call("episode", "call", reason="context_length_exceeded")
    with pytest.raises(ValueError, match="new call id"):
        await store.begin_call("episode", "call")


async def test_synthetic_length_completion_is_not_a_token_record(tmp_path):
    store = TokenCaptureStore(tmp_path)
    context = CaptureContext("episode", "overflow", store)
    token = set_token_sink(context)
    try:
        await register_call_intent()
        await record_call_rejection(reason="context_length_exceeded")
        await capture_tokens({"choices": [{"finish_reason": "length", "message": {"content": ""}}]})
    finally:
        reset_token_sink(token)
    snapshot = await store.freeze("episode")
    assert context.rejected_without_generation
    assert not context.committed
    assert not snapshot.incomplete
    assert not snapshot.entries
    # An episode with no successful generations still has nothing to train on.
    assert trajectories_for_rollout("episode", [tmp_path], builder="independent_calls")["mask_sample"]


@pytest.mark.parametrize("failure", ["missing_extension", "write_failure", "already_committed"])
async def test_rejection_hook_fails_closed(tmp_path, monkeypatch, failure):
    store = TokenCaptureStore(tmp_path)
    context = CaptureContext("episode", "overflow", store)
    if failure == "missing_extension":
        monkeypatch.setattr(store, "reject_call", None)
    elif failure == "write_failure":
        monkeypatch.setattr(store, "reject_call", AsyncMock(side_effect=OSError("cannot persist")))
    else:
        await store.put(entry("overflow"))
        context.committed = True
    token = set_token_sink(context)
    try:
        await register_call_intent()
        await record_call_rejection(reason="context_length_exceeded")
    finally:
        reset_token_sink(token)
    assert not context.rejected_without_generation
    assert (await store.freeze("episode")).incomplete


@pytest.mark.parametrize("rejected", [False, True])
async def test_actual_tokens_at_length_limit_are_captured_or_flag_a_conflict(tmp_path, rejected):
    store = TokenCaptureStore(tmp_path)
    context = CaptureContext("episode", "length", store)
    token = set_token_sink(context)
    try:
        await register_call_intent()
        if rejected:
            await record_call_rejection(reason="context_length_exceeded")
        await capture_tokens(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "generated up to the limit",
                            "prompt_token_ids": [1, 2],
                            "generation_token_ids": [3],
                            "generation_log_probs": [-0.5],
                        },
                    }
                ]
            },
            request_messages=[{"role": "user", "content": "input"}],
        )
    finally:
        reset_token_sink(token)
    snapshot = await store.freeze("episode")
    assert snapshot.incomplete is rejected
    assert context.committed is not rejected
    if not rejected:
        assert snapshot.entries[0].generation_token_ids == [3]
    else:
        assert not snapshot.entries
