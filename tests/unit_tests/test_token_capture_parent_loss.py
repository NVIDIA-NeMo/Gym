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

"""A rollout's token records deleted under a live rollout (2026-10-19, "delta record has no indexed parent").

Two ``/run`` handlers can share one rollout id (Gym re-sends ``/run`` with the same body after a
dropped connection). The first to finish deletes the rollout's records; the other handler's
in-flight call resolved its parent before the deletion and used to commit an orphan delta into
the recreated file, after which every later resolution of that rollout raised. These tests pin
the three defenses: the store refuses the orphan, the sink re-roots the call from its full
prompt so the next continuation stays token-contiguous, the lineage store reports a pre-existing
orphan as unresolved instead of raising, and the cleanup helper leaves records that end with
another handler's call.
"""

import json

import orjson
import pytest

from nemo_gym.token_id_capture.lineage import FileLineageStore, stamp_continuation
from nemo_gym.token_id_capture.records import (
    PARENT_RECORDS_LOST_REASON,
    ParentRecordsLostError,
    ParentResolutionStatus,
    TokenEntry,
    stamp_lineage,
)
from nemo_gym.token_id_capture.sink import CaptureContext, commit_entry, reset_token_sink, resolve_parent, set_token_sink
from nemo_gym.token_id_capture.store import TokenCaptureStore


ROLLOUT = "task-7-g3"


def _root_entry(call_id="call-1", response_id="chatcmpl-1"):
    request = [{"role": "user", "content": "hello"}]
    entry = TokenEntry(
        rollout_id=ROLLOUT, model_call_id=call_id, prompt_token_ids=[1, 2], generation_token_ids=[3],
        generation_log_probs=[-0.1], output_items=[{"role": "assistant", "content": "hi"}], response_id=response_id,
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    stamp_continuation(entry, request)
    return entry, request


def _delta_entry(parent_call_id, call_id="call-2", response_id="chatcmpl-2"):
    entry = TokenEntry(
        rollout_id=ROLLOUT, model_call_id=call_id, prompt_token_ids=[4], generation_token_ids=[5],
        generation_log_probs=[-0.1], output_items=[{"role": "assistant", "content": "more"}],
        parent_call_id=parent_call_id, parent_resolution=ParentResolutionStatus.RESOLVED, prompt_is_delta=True,
        response_id=response_id,
    )
    stamp_lineage(entry, parent_call_id, parent_resolution=ParentResolutionStatus.RESOLVED, cumulative=[1, 2, 3, 4, 5])
    return entry


def _rows(store, rollout_id=ROLLOUT):
    path = store.path_for(rollout_id)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# ---------------------------------------------------------------------------
# Store: never append an orphan delta
# ---------------------------------------------------------------------------


def test_store_refuses_a_delta_whose_parent_is_gone(tmp_path):
    store = TokenCaptureStore(tmp_path)
    root, _ = _root_entry()
    store.append(root)
    store.append(_delta_entry("call-1"))  # the normal case still works
    assert [r["model_call_id"] for r in _rows(store)] == ["call-1", "call-2"]

    store.delete(ROLLOUT)  # the other handler's cleanup
    with pytest.raises(ParentRecordsLostError, match="no longer among the rollout's records"):
        store.append(_delta_entry("call-1", call_id="call-3"))
    assert _rows(store) == []  # nothing was written

    # A re-rooted record (full prompt, no parent) is accepted into the recreated file.
    reroot, _ = _root_entry(call_id="call-3", response_id="chatcmpl-3")
    store.append(reroot)
    assert [r["model_call_id"] for r in _rows(store)] == ["call-3"]


# ---------------------------------------------------------------------------
# Sink: re-root from the full prompt, keep the next continuation contiguous
# ---------------------------------------------------------------------------


async def test_sink_reroots_a_call_whose_parent_vanished_before_commit(tmp_path, caplog):
    store = TokenCaptureStore(tmp_path)
    resolver = FileLineageStore(tmp_path)
    root, request = _root_entry()
    store.append(root)
    history = request + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "go on"}]

    context = CaptureContext(rollout_id=ROLLOUT, model_call_id="call-2", token_sink=store, lineage_store=resolver, delta_records=True)
    token = set_token_sink(context)
    try:
        await resolve_parent(history)
        assert context.parent_resolution.status == ParentResolutionStatus.RESOLVED
        assert context.parent_resolution.match.cumulative_token_ids == (1, 2, 3)

        store.delete(ROLLOUT)  # between parent resolution and commit: the other handler finished

        # Generation-time prompt = the parent's exact tokens + the new user turn.
        entry = TokenEntry(
            rollout_id=ROLLOUT, model_call_id="call-2", prompt_token_ids=[1, 2, 3, 4], generation_token_ids=[5],
            generation_log_probs=[-0.1], output_items=[{"role": "assistant", "content": "more"}], response_id="chatcmpl-2",
        )
        stamp_continuation(entry, history)
        with caplog.at_level("WARNING"):
            await commit_entry(entry)
    finally:
        reset_token_sink(token)

    assert context.committed is True
    (row,) = _rows(store)
    assert row["model_call_id"] == "call-2" and row["prompt_is_delta"] is False
    assert row["prompt_token_ids"] == [1, 2, 3, 4] and row["parent_call_id"] is None
    assert row["parent_resolution"] == "unresolved" and row["parent_resolution_reason"] == PARENT_RECORDS_LOST_REASON
    assert row["cum_len"] == 5
    assert not store.is_incomplete(ROLLOUT)
    assert any("re-rooting the call" in message for message in caplog.messages)

    # The next continuation resolves to the re-rooted call with the full, contiguous tokens.
    resolution = await resolver.resolve(ROLLOUT, history + [{"role": "assistant", "content": "more"}, {"role": "user", "content": "next"}])
    assert resolution.status == ParentResolutionStatus.RESOLVED
    assert resolution.match.model_call_id == "call-2"
    assert resolution.match.cumulative_token_ids == (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Lineage store: a pre-existing orphan is unresolved, not an exception
# ---------------------------------------------------------------------------


async def test_lineage_reports_an_orphan_delta_as_unresolved(tmp_path, caplog):
    # Written directly (an old store could append this); the index has no "call-1".
    orphan = _delta_entry("call-1")
    stamp_continuation(orphan, [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "go on"}])
    path = TokenCaptureStore(tmp_path).path_for(ROLLOUT)
    path.write_bytes(orjson.dumps(orphan.model_dump(mode="json")) + b"\n")

    resolver = FileLineageStore(tmp_path)
    continuation = [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "go on"},
        {"role": "assistant", "content": "more"}, {"role": "user", "content": "next"},
    ]
    with caplog.at_level("WARNING"):
        resolution = await resolver.resolve(ROLLOUT, continuation)
    assert resolution.status == ParentResolutionStatus.UNRESOLVED
    assert resolution.reason == PARENT_RECORDS_LOST_REASON
    assert any("has no indexed parent" in message for message in caplog.messages)

    # Through the sink: an unresolved decision with the reason, no traceback.
    context = CaptureContext(rollout_id=ROLLOUT, model_call_id="call-3", token_sink=None, lineage_store=resolver)
    token = set_token_sink(context)
    try:
        await resolve_parent(continuation)
    finally:
        reset_token_sink(token)
    assert context.parent_resolution.status == ParentResolutionStatus.UNRESOLVED
    assert context.parent_resolution.reason == PARENT_RECORDS_LOST_REASON


# ---------------------------------------------------------------------------
# Cleanup: keep records another handler is still appending to
# ---------------------------------------------------------------------------


def test_delete_unless_foreign_keeps_records_that_end_with_another_handlers_call(tmp_path):
    store = TokenCaptureStore(tmp_path)
    root, _ = _root_entry(response_id="chatcmpl-1")
    store.append(root)
    store.append(_delta_entry("call-1", response_id="chatcmpl-2"))
    assert store._last_response_id(ROLLOUT) == "chatcmpl-2"

    assert store.delete_unless_foreign(ROLLOUT, "chatcmpl-1") is False  # ours was call 1; call 2 is someone else's
    assert len(_rows(store)) == 2 and store.state_path_for(ROLLOUT).exists()

    assert store.delete_unless_foreign(ROLLOUT, "chatcmpl-2") is True
    assert not store.path_for(ROLLOUT).exists() and not store.state_path_for(ROLLOUT).exists()
    assert store.delete_unless_foreign(ROLLOUT, "chatcmpl-2") is True  # nothing left: idempotent

    # Unknown last call (no policy call made by this handler): unconditional, as before.
    store.append(_root_entry(response_id="chatcmpl-9")[0])
    assert store.delete_unless_foreign(ROLLOUT, None) is True and not store.path_for(ROLLOUT).exists()


def test_last_response_id_reads_a_tail_longer_than_the_first_window(tmp_path):
    store = TokenCaptureStore(tmp_path)
    big, _ = _root_entry(response_id="chatcmpl-big")
    big.prompt_token_ids = list(range(400_000))  # ~2.5 MB line, beyond the 1 MiB first window
    stamp_lineage(big, None, parent_resolution=ParentResolutionStatus.ROOT)
    store.append(big)
    assert store._last_response_id(ROLLOUT) == "chatcmpl-big"
    assert store._last_response_id("never-seen") is None


async def test_lineage_cursor_survives_a_recreated_file_that_reuses_the_inode(tmp_path):
    """After delete + recreate the inode number can repeat; the cursor must not read a stale offset."""
    store = TokenCaptureStore(tmp_path)
    resolver = FileLineageStore(tmp_path)
    continuation = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "next"}]
    root, _ = _root_entry(call_id="first-incarnation")
    store.append(root)
    assert (await resolver.resolve(ROLLOUT, continuation)).match.model_call_id == "first-incarnation"
    old_inode = store.path_for(ROLLOUT).stat().st_ino

    store.delete(ROLLOUT)
    reroot, _ = _root_entry(call_id="second-incarnation", response_id="chatcmpl-2")
    reroot.prompt_token_ids = [7, 8, 9, 10, 11, 12]  # longer first line than before: old offset is inside it
    stamp_lineage(reroot, None, parent_resolution=ParentResolutionStatus.ROOT)
    store.append(reroot)
    # Force the stale-cursor shape the inode-reuse case produces: same inode, old offset.
    cursor, refs, lineage = resolver._cache[ROLLOUT]
    resolver._cache[ROLLOUT] = ((store.path_for(ROLLOUT).stat().st_ino, cursor[1], cursor[2]), refs, lineage)

    resolution = await resolver.resolve(ROLLOUT, continuation)
    assert resolution.status == ParentResolutionStatus.RESOLVED
    assert resolution.match.model_call_id == "second-incarnation"
    assert resolution.match.cumulative_token_ids == (7, 8, 9, 10, 11, 12, 3)
    del old_inode
