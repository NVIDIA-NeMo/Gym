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
"""Whole-turn fallback preserves durable token custody and rejects unrecorded output."""

from copy import deepcopy

import pytest

from nemo_gym.token_id_capture import stamp_lineage
from nemo_gym.token_id_capture.fingerprint import FINGERPRINT_VERSION, tool_call_turn_fingerprint
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore, stamp_continuation
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry
from nemo_gym.token_id_capture.staging.digest import EMPTY_EXTRAS_DIGEST, compute_chain_hash, hash_token_ids
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit
from nemo_gym.token_id_capture.store import TokenCaptureStore


CONTEXT = [{"role": "user", "content": "Inspect the repository."}]
TOKENS = [11, 12, 101]


def _turn(arguments: str = '{"command":"cd /repo && ls"}') -> list[dict]:
    return [
        {"role": "assistant", "content": "I will inspect it."},
        {"type": "function_call", "call_id": "call_0", "name": "Bash", "arguments": arguments},
        {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"},
    ]


def _request(*, rewritten: bool = True) -> list[dict]:
    turn = _turn('{"command":"ls"}') if rewritten else _turn()
    return (
        deepcopy(CONTEXT)
        + turn
        + [
            {"type": "function_call_output", "call_id": "call_0", "output": "files"},
            {"type": "function_call_output", "call_id": "call_1", "output": "contents"},
            {"role": "user", "content": "continue"},
        ]
    )


def _commit(call_id: str, output: list[dict], tokens: list[int], version: int) -> CaptureLedgerCommit:
    record = CallRecord(
        model_call_id=call_id,
        staging_key=f"r/{call_id}",
        weight_version=7,
        prev_len=0,
        delta_len=len(tokens),
        cum_len=len(tokens),
        digest="a" * 64,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        mode="text",
        chain_hash=compute_chain_hash(None, tokens),
        cumulative_hash=hash_token_ids(tokens),
        response_id=f"response-{call_id}",
        fingerprint_version=version,
    )
    return CaptureLedgerCommit(
        rollout_id="r", record=record, request_items=CONTEXT, response_items=output, staging_chain=(f"r/{call_id}",)
    )


@pytest.fixture(params=["local-file", "custody-file", "local-memory", "custody-memory"])
def stores(request, tmp_path):
    kind = request.param
    if kind == "local-file":
        writer, reader = TokenCaptureStore(tmp_path), FileLineageStore(tmp_path)
    elif kind == "custody-file":
        writer, reader = FileLineageStore(tmp_path), FileLineageStore(tmp_path)
    else:
        writer = reader = InMemoryLineageStore()
    return kind, writer, reader


async def _publish(
    stores,
    call_id: str = "a",
    *,
    output: list[dict] | None = None,
    tokens: list[int] | None = None,
    version: int = FINGERPRINT_VERSION,
) -> None:
    kind, writer, _ = stores
    output = _turn() if output is None else output
    tokens = TOKENS if tokens is None else tokens
    if kind.startswith("custody"):
        await writer.record(_commit(call_id, output, tokens, version))
    else:
        entry = TokenEntry(
            rollout_id="r",
            model_call_id=call_id,
            prompt_token_ids=tokens[:-1],
            generation_token_ids=tokens[-1:],
            generation_log_probs=[-0.1],
            output_items=output,
            response_id=f"response-{call_id}",
        )
        stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
        stamp_continuation(entry, CONTEXT)
        entry.fingerprint_version = version
        await writer.put(entry)


@pytest.mark.asyncio
async def test_rewritten_turn_resolves_original_tokens_or_custody(stores) -> None:
    kind, _, reader = stores
    # Populate the reader's empty cache before another handle commits.
    assert (await reader.resolve("r", _request())).match is None
    await _publish(stores)
    result = await reader.resolve("r", _request())
    assert result.status == ParentResolutionStatus.RESOLVED
    assert result.reason == "tool_call_turn"
    assert result.match.model_call_id == "a"
    if kind.startswith("custody"):
        assert result.match.cumulative_token_ids == ()
        assert result.match.staging_chain == ("r/a",)
        assert result.match.prev_len == len(TOKENS)
        assert result.match.digest == hash_token_ids(TOKENS)
        assert result.match.chain_hash == compute_chain_hash(None, TOKENS)
    else:
        assert result.match.cumulative_token_ids == tuple(TOKENS)
        if kind == "local-file":
            assert reader._cache["r"][2].by_call_id["a"].cum_tokens is None


@pytest.mark.asyncio
async def test_strict_same_id_sibling_wins_but_rewritten_siblings_are_ambiguous(stores) -> None:
    await _publish(stores)
    await _publish(stores, "b", output=_turn('{"command":"pytest"}'), tokens=[11, 12, 102])
    reader = stores[2]
    exact = await reader.resolve("r", _request(rewritten=False))
    assert exact.status == ParentResolutionStatus.RESOLVED
    assert exact.match.model_call_id == "a"
    assert exact.reason == ""
    rewritten = await reader.resolve("r", _request())
    assert rewritten.status == ParentResolutionStatus.UNRESOLVED
    assert rewritten.match is None
    assert rewritten.reason == "ambiguous"


@pytest.mark.asyncio
@pytest.mark.parametrize("rewritten", [False, True])
async def test_identical_retries_collapse_consistently(stores, rewritten) -> None:
    await _publish(stores, "z")
    await _publish(stores, "a")
    result = await stores[2].resolve("r", _request(rewritten=rewritten))
    assert result.status == ParentResolutionStatus.RESOLVED
    assert result.match.model_call_id == "a"
    assert result.reason == ("tool_call_turn" if rewritten else "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "context",
        "text",
        "name",
        "id",
        "namespace",
        "add",
        "remove",
        "reorder",
        "empty_message",
        "duplicate_id",
        "missing_id",
        "inserted_tool",
        "inserted_text",
    ],
)
async def test_whole_turn_fallback_rejects_other_changes(stores, change) -> None:
    await _publish(stores)
    request = _request()
    if change == "context":
        request[0]["content"] = "different context"
    elif change == "text":
        request[1]["content"] = "different answer"
    elif change in {"name", "namespace"}:
        request[2][change] = "different"
    elif change == "id":
        request[2]["call_id"] = "different"
    elif change == "add":
        request.insert(4, {**request[2], "call_id": "new"})
    elif change == "remove":
        request.pop(3)
    elif change == "reorder":
        request[2], request[3] = request[3], request[2]
    elif change == "empty_message":
        request.insert(4, {"role": "assistant", "content": ""})
    elif change == "duplicate_id":
        request[3]["call_id"] = request[2]["call_id"]
    elif change == "missing_id":
        request[2].pop("call_id")
    elif change == "inserted_tool":
        request[1:1] = [
            {"type": "function_call", "call_id": "extra", "name": "Bash", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "extra", "output": "extra result"},
        ]
    elif change == "inserted_text":
        request[1:1] = [{"role": "assistant", "content": "uncaptured"}, {"role": "user", "content": "next"}]
    result = await stores[2].resolve("r", request)
    assert result.status == ParentResolutionStatus.UNRESOLVED
    assert result.match is None


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [0, FINGERPRINT_VERSION - 1, FINGERPRINT_VERSION + 1])
async def test_incompatible_versions_do_not_resolve_then_refresh_current_rows(stores, version) -> None:
    await _publish(stores, "old", version=version)
    for rewritten in (False, True):
        result = await stores[2].resolve("r", _request(rewritten=rewritten))
        assert result.status == ParentResolutionStatus.UNRESOLVED
        assert result.match is None
    await _publish(stores, "current")
    result = await stores[2].resolve("r", _request())
    assert result.match.model_call_id == "current"
    assert result.reason == "tool_call_turn"


@pytest.mark.asyncio
async def test_interleaved_reasoning_is_part_of_the_whole_turn(stores) -> None:
    output = _turn()
    output.insert(2, {"type": "reasoning", "summary": []})
    await _publish(stores, output=output)
    request = _request()
    request.insert(3, {"type": "reasoning", "summary": []})
    result = await stores[2].resolve("r", request)
    assert result.match.model_call_id == "a"
    assert result.reason == "tool_call_turn"


@pytest.mark.asyncio
async def test_custody_column_survives_restart_and_checkpoint_import(tmp_path) -> None:
    writer = FileLineageStore(tmp_path / "source")
    commit = _commit("a", _turn(), TOKENS, FINGERPRINT_VERSION)
    await writer.record(commit)
    rows = writer.export_rows("r")
    assert rows[0]["tool_call_turn_fingerprint"] == tool_call_turn_fingerprint(_turn())
    assert "cumulative_token_ids" not in rows[0]
    assert "output_items" not in rows[0]
    for reader in (FileLineageStore(tmp_path / "source"), FileLineageStore(tmp_path / "restored")):
        reader.import_rows("r", rows)
        result = await reader.resolve("r", _request())
        assert result.reason == "tool_call_turn"
        assert result.match.staging_chain == ("r/a",)
        assert result.match.cumulative_token_ids == ()


@pytest.mark.asyncio
async def test_legacy_custody_rows_are_strict_only_and_replay_is_idempotent(tmp_path) -> None:
    writer = FileLineageStore(tmp_path)
    reader = FileLineageStore(tmp_path)
    commit = _commit("a", _turn(), TOKENS, FINGERPRINT_VERSION)
    await writer.record(commit)
    rows = writer.export_rows("r")
    rows[0].pop("tool_call_turn_fingerprint")
    writer.import_rows("r", rows)
    await writer.record(commit)
    assert len(writer.export_rows("r")) == 1
    assert (await reader.resolve("r", _request(rewritten=False))).match.model_call_id == "a"
    assert (await reader.resolve("r", _request())).match is None
    # The old cached row does not hide a new schema row appended by another worker.
    await writer.record(_commit("b", _turn(), TOKENS, FINGERPRINT_VERSION))
    result = await reader.resolve("r", _request())
    assert result.match.model_call_id == "b"
    assert result.reason == "tool_call_turn"
    assert len((await reader.manifest("r"))["records"]) == 2


@pytest.mark.asyncio
async def test_unversioned_custody_row_cannot_use_persisted_fallback_key(tmp_path) -> None:
    writer = FileLineageStore(tmp_path)
    await writer.record(_commit("a", _turn(), TOKENS, FINGERPRINT_VERSION))
    rows = writer.export_rows("r")
    rows[0].pop("fingerprint_version")
    writer.import_rows("r", rows)
    result = await FileLineageStore(tmp_path).resolve("r", _request())
    assert result.status == ParentResolutionStatus.UNRESOLVED
    assert result.match is None
