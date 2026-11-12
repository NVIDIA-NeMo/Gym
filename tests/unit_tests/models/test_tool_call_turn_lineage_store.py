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

import json
from copy import deepcopy

import pytest

from nemo_gym.models.token_capture.fingerprint import (
    FINGERPRINT_VERSION,
    echoed_tool_call_turn_fingerprints,
    tool_call_turn_fingerprint,
)
from nemo_gym.models.token_capture.lineage import FileLineageStore, InMemoryLineageStore, stamp_continuation
from nemo_gym.models.token_capture.records import ParentResolutionStatus, TokenEntry, stamp_lineage
from nemo_gym.models.token_capture.staging.digest import EMPTY_EXTRAS_DIGEST, compute_chain_hash, hash_token_ids
from nemo_gym.models.token_capture.staging.records import CallRecord, CaptureLedgerCommit
from nemo_gym.models.token_capture.store import TokenCaptureStore


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
        capture_key="r", record=record, request_items=CONTEXT, response_items=output, staging_chain=(f"r/{call_id}",)
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
            capture_key="r",
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


def _unavailable_tool_placeholder(arguments: dict) -> list[dict]:
    """The request as OpenCode sends it after the model called a tool OpenCode does not offer: the call keeps its
    served ID and becomes a call to OpenCode's `invalid` tool."""
    request = _request()
    request[3] = {**request[3], "name": "invalid", "arguments": json.dumps(arguments)}
    return request


@pytest.mark.asyncio
async def test_opencode_unavailable_tool_placeholder_resolves_to_the_served_call(stores) -> None:
    await _publish(stores)
    request = _unavailable_tool_placeholder(
        {"tool": "Read", "error": "Model tried to call unavailable tool 'Read'. Available tools: bash, invalid."}
    )
    result = await stores[2].resolve("r", request)
    assert result.status == ParentResolutionStatus.RESOLVED
    assert result.match.model_call_id == "a"
    assert result.reason == "tool_call_turn"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"tool": "Bash", "error": "names another served tool"},
        {"tool": "Read"},
        {"tool": "Read", "error": "unavailable", "extra": "field"},
        {"tool": "", "error": "unavailable"},
        {"tool": "Read", "error": None},
    ],
)
async def test_only_the_exact_placeholder_of_the_served_tool_resolves(stores, arguments) -> None:
    await _publish(stores)
    result = await stores[2].resolve("r", _unavailable_tool_placeholder(arguments))
    assert result.status == ParentResolutionStatus.UNRESOLVED
    assert result.match is None


def test_placeholder_keeps_the_served_turn_key_and_other_renames_do_not() -> None:
    served = _turn()
    placeholder = deepcopy(served)
    placeholder[2] = {**placeholder[2], "name": "invalid", "arguments": '{"error":"unavailable","tool":"Read"}'}
    renamed = deepcopy(served)
    renamed[2] = {**renamed[2], "name": "invalid", "arguments": "{}"}
    assert tool_call_turn_fingerprint(
        placeholder, read_placeholders=frozenset({"call_1"})
    ) == tool_call_turn_fingerprint(served)
    assert tool_call_turn_fingerprint(renamed, read_placeholders=frozenset({"call_1"})) != tool_call_turn_fingerprint(
        served
    )
    # Keys of served output never read placeholders.
    assert tool_call_turn_fingerprint(placeholder) != tool_call_turn_fingerprint(served)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("Invalid", '{"tool": "Read", "error": "unavailable"}'),
        ("invalid_tool", '{"tool": "Read", "error": "unavailable"}'),
        ("invalid", '["Read", "unavailable"]'),
        ("invalid", "not json"),
        ("invalid", '{"tool": 1, "error": "unavailable"}'),
        ("invalid", '{"tool": "", "error": "unavailable"}'),
        ("invalid", '{"tool": "Read", "error": 1}'),
    ],
)
def test_only_the_exact_placeholder_shape_keeps_the_served_turn_key(name: str, arguments: str) -> None:
    served = _turn()
    echoed = deepcopy(served)
    echoed[2] = {**echoed[2], "name": name, "arguments": arguments}
    assert tool_call_turn_fingerprint(served) not in echoed_tool_call_turn_fingerprints(echoed)


def test_placeholder_in_a_chat_tool_call_keeps_the_served_turn_key() -> None:
    # OpenCode sends Chat Completions requests, where a call is nested in the assistant message's tool_calls.
    def chat_turn(name: str, arguments: str) -> list[dict]:
        call = {"id": "call-1", "type": "function", "function": {"name": name, "arguments": arguments}}
        return [{"role": "assistant", "content": None, "tool_calls": [call]}]

    served = chat_turn("Read", '{"filePath": "/app/notes.md"}')
    echoed = chat_turn(
        "invalid", json.dumps({"tool": "Read", "error": "Model tried to call unavailable tool 'Read'."})
    )
    renamed = chat_turn("Write", '{"filePath": "/app/notes.md"}')
    assert tool_call_turn_fingerprint(served)
    assert tool_call_turn_fingerprint(served) in echoed_tool_call_turn_fingerprints(echoed)
    assert tool_call_turn_fingerprint(served) not in echoed_tool_call_turn_fingerprints(renamed)


def _placeholder_shaped(error: str) -> list[dict]:
    """A model turn whose own call has the placeholder's shape: the model called `invalid` itself."""
    turn = _turn()
    turn[2] = {**turn[2], "name": "invalid", "arguments": json.dumps({"tool": "Read", "error": error})}
    return turn


@pytest.mark.asyncio
async def test_a_served_placeholder_shaped_call_resolves_when_its_echo_rewrites_arguments(stores) -> None:
    await _publish(stores, output=_placeholder_shaped("bad"))
    request = _request()
    request[3] = _placeholder_shaped("bad, as the harness re-serialized it")[2]
    result = await stores[2].resolve("r", request)
    assert result.status == ParentResolutionStatus.RESOLVED
    assert result.match.model_call_id == "a"
    assert result.reason == "tool_call_turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True], ids=["rewritten call first", "own invalid call first"])
async def test_a_turn_mixing_a_rewritten_call_and_an_own_invalid_call_resolves(stores, reverse) -> None:
    # OpenCode rewrote the model's Bash call into its placeholder, and the model also called `invalid` itself.
    # Each placeholder-shaped call is read on its own, so the turn still matches what was served.
    served = _placeholder_shaped("bad")
    echoed = deepcopy(served)
    echoed[1] = {**echoed[1], "name": "invalid", "arguments": json.dumps({"tool": "Bash", "error": "unavailable"})}
    echoed[2] = {**echoed[2], "arguments": json.dumps({"tool": "Read", "error": "rewritten"})}
    if reverse:
        served[1], served[2] = served[2], served[1]
        echoed[1], echoed[2] = echoed[2], echoed[1]
    await _publish(stores, output=served)
    result = await stores[2].resolve("r", deepcopy(CONTEXT) + echoed + _request()[4:])
    assert result.status == ParentResolutionStatus.RESOLVED
    assert result.match.model_call_id == "a"
    assert result.reason == "tool_call_turn"


def _read_call(call_id: str) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": "Read", "arguments": "{}"}


def _invalid_call(call_id: str, error: str) -> dict:
    """A call with the placeholder's shape: OpenCode's rewrite of `Read`, or the model's own `invalid` call."""
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "invalid",
        "arguments": json.dumps({"tool": "Read", "error": error}),
    }


def _calls_request(calls: list[dict]) -> list[dict]:
    return (
        deepcopy(CONTEXT)
        + [{"role": "assistant", "content": "Read them all."}, *calls]
        + [{"type": "function_call_output", "call_id": call["call_id"], "output": "ok"} for call in calls]
        + [{"role": "user", "content": "continue"}]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("published", ["a", "b", "both"])
async def test_every_placeholder_reading_competes_before_a_parent_is_chosen(stores, published) -> None:
    # Echoed as six placeholders, the turn matches `a` (six Read calls) when all are read, and `b` (one Read call,
    # then five calls the model made to `invalid` itself) when only the first is read. Both must compete.
    ids = [f"c{index}" for index in range(6)]
    turns = {
        "a": [_read_call(call_id) for call_id in ids],
        "b": [_read_call(ids[0])] + [_invalid_call(call_id, "bad") for call_id in ids[1:]],
    }
    for name, tokens in (("a", [11, 12, 101]), ("b", [11, 12, 102])):
        if published in (name, "both"):
            await _publish(
                stores, name, output=[{"role": "assistant", "content": "Read them all."}, *turns[name]], tokens=tokens
            )
    result = await stores[2].resolve("r", _calls_request([_invalid_call(call_id, "rewritten") for call_id in ids]))
    if published == "both":
        assert result.status == ParentResolutionStatus.UNRESOLVED
        assert result.reason == "ambiguous"
    else:
        assert result.status == ParentResolutionStatus.RESOLVED
        assert result.match.model_call_id == published
        assert result.reason == "tool_call_turn"


@pytest.mark.asyncio
async def test_more_placeholder_shaped_calls_than_the_cap_leave_only_the_strict_match(stores) -> None:
    # Reading 7 placeholder-shaped calls every way is too many keys, so the fallback abstains rather than decide
    # uniqueness from some of the readings. An exact echo still resolves through the strict fingerprint.
    served = [_read_call(f"c{index}") for index in range(7)]
    await _publish(stores, output=[{"role": "assistant", "content": "Read them all."}, *served])
    rewritten = await stores[2].resolve(
        "r", _calls_request([_invalid_call(call["call_id"], "rewritten") for call in served])
    )
    assert rewritten.status == ParentResolutionStatus.UNRESOLVED
    assert rewritten.match is None
    exact = await stores[2].resolve("r", _calls_request(served))
    assert exact.status == ParentResolutionStatus.RESOLVED
    assert exact.match.model_call_id == "a"
    assert echoed_tool_call_turn_fingerprints(_calls_request(served)[1:])


@pytest.mark.asyncio
async def test_a_served_placeholder_shaped_call_keeps_its_stored_key_across_replay(tmp_path) -> None:
    # Stored keys never read placeholders, so a ledger written before this mapping existed holds the same key, and
    # replaying its commit after a checkpoint import stays a no-op instead of a conflicting record.
    output = _placeholder_shaped("bad")
    commit = _commit("a", output, TOKENS, FINGERPRINT_VERSION)
    writer = FileLineageStore(tmp_path)
    await writer.record(commit)
    [row] = writer.export_rows("r")
    assert row["tool_call_turn_fingerprint"] == tool_call_turn_fingerprint(output)
    assert row["tool_call_turn_fingerprint"] != tool_call_turn_fingerprint(
        output, read_placeholders=frozenset({"call_1"})
    )
    restored = FileLineageStore(tmp_path / "restored")
    restored.import_rows("r", [row])
    await restored.record(commit)
    assert restored.export_rows("r") == [row]


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
    assert rows[0].get("output_items") is None
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


def _repaired_request(name: str, arguments: str = '{"command":"cd /repo && ls"}') -> list[dict]:
    """The request after OpenCode's tool-call repair renamed the served ``Bash`` call ``call_0``."""
    request = _request(rewritten=False)
    request[2] = {"type": "function_call", "call_id": "call_0", "name": name, "arguments": arguments}
    return request


@pytest.mark.asyncio
async def test_a_call_that_opencode_lowercased_resolves_to_the_served_call(stores) -> None:
    # OpenCode renames a call whose lowercase name is a known tool to that name, in every later request.
    await _publish(stores)
    result = await stores[2].resolve("r", _repaired_request("bash"))
    assert result.status == ParentResolutionStatus.RESOLVED
    assert result.reason == "tool_call_turn"
    assert result.match.model_call_id == "a"


@pytest.mark.asyncio
async def test_a_call_renamed_to_another_tool_stays_unresolved(stores) -> None:
    await _publish(stores)
    result = await stores[2].resolve("r", _repaired_request("grep"))
    assert result.status == ParentResolutionStatus.UNRESOLVED


def test_the_poison_log_names_only_the_tools_of_the_last_model_turn() -> None:
    from nemo_gym.models.token_capture.sink import _trailing_tool_names

    assert _trailing_tool_names(_repaired_request("bash")) == ["bash", "Read"]
    assert _trailing_tool_names(None) == []
    assert _trailing_tool_names([{"role": "user", "content": "hi"}]) == []
