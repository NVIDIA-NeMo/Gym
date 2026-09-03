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

"""Verify that an external token-capture transport follows Gym's protocols.

A framework transport can replace Gym's file store.
For example, NeMo-RL can carry records through a transfer queue.
An incorrect implementation can lose records or make a committed parent invisible to another client.
``run_conformance`` checks these behaviors directly.
Factories must return fresh client instances over the same shared backend.
The checks use fresh instances to verify cross-client visibility.
Each check uses its own rollout id, so failed checks cannot poison later ones.
This module avoids FastAPI, Ray, Torch, and aiohttp imports.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from nemo_gym.token_id_capture.lineage import stamp_continuation
from nemo_gym.token_id_capture.protocols import LineageStore, TokenSink, TokenSource
from nemo_gym.token_id_capture.records import (
    ParentResolutionStatus,
    TokenEntry,
    cumulative_tokens,
    stamp_lineage,
)


class ConformanceError(AssertionError):
    """One named contract check failed."""

    def __init__(self, check_name: str, detail: str) -> None:
        self.check_name = check_name
        self.detail = detail
        super().__init__(f"{check_name}: {detail}")


def _require(condition: bool, check_name: str, detail: str) -> None:
    if not condition:
        raise ConformanceError(check_name, detail)


def _make_entry(
    rollout_id: str,
    model_call_id: str,
    *,
    prompt: list[int],
    generation: list[int],
    request_items: list[dict],
    text: str,
) -> TokenEntry:
    """Build a committed entry with the metadata that ``capture_tokens`` records."""
    entry = TokenEntry(
        rollout_id=rollout_id,
        model_call_id=model_call_id,
        model="conformance-model",
        prompt_token_ids=list(prompt),
        generation_token_ids=list(generation),
        generation_log_probs=[-0.1] * len(generation),
        output_items=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        token_item_index=0,
        # Fixed so an idempotent retry can resend byte-identical payloads.
        created_at=1700000000.0,
    )
    stamp_continuation(entry, list(request_items))
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    return entry


def _identical_retry(entry: TokenEntry) -> TokenEntry:
    """Simulate a retry that resends the same serialized entry."""
    return TokenEntry.model_validate(entry.model_dump(mode="json"))


_REQUEST = [{"role": "user", "content": "What is the weather in Paris?"}]


async def run_conformance(
    sink_factory: Callable[[], TokenSink],
    source_factory: Callable[[], TokenSource],
    lineage_factory: Callable[[], LineageStore] | None = None,
    *,
    rollout_id: str = "conformance-rollout",
) -> list[str]:
    """Run the ordered contract checks and return the names that passed.

    Raise ``ConformanceError`` on the first failure.
    Lineage checks are skipped without a ``lineage_factory``.
    The ``begin_call`` check is skipped when the sink lacks the extension.
    """
    passed: list[str] = []
    closables: list = []

    def sink() -> TokenSink:
        instance = sink_factory()
        closables.append(instance)
        return instance

    def source() -> TokenSource:
        instance = source_factory()
        closables.append(instance)
        return instance

    def lineage() -> LineageStore:
        assert lineage_factory is not None
        instance = lineage_factory()
        closables.append(instance)
        return instance

    checks: list[tuple[str, Callable[[str], Awaitable[None]]]] = [
        ("put_then_freeze_visibility", lambda r: _check_put_then_freeze(sink(), source(), r)),
        ("idempotent_reput", lambda r: _check_idempotent_reput(sink(), source(), r)),
        ("conflicting_reput", lambda r: _check_conflicting_reput(sink(), source(), r)),
        ("mark_incomplete_durability", lambda r: _check_mark_incomplete(sink(), source, r)),
        ("freeze_idempotency", lambda r: _check_freeze_idempotency(sink(), source, r)),
        ("post_freeze_write_safety", lambda r: _check_post_freeze_write(sink(), source(), r)),
        ("conditional_retirement", lambda r: _check_conditional_retirement(sink(), source(), r)),
    ]
    if lineage_factory is not None:
        checks.append(
            ("lineage_visibility", lambda r: _check_lineage_visibility(sink(), lineage, r, fresh_client=False))
        )
        checks.append(
            (
                "fresh_client_lineage_visibility",
                lambda r: _check_lineage_visibility(sink(), lineage, r, fresh_client=True),
            )
        )
    probe_sink = sink()
    if getattr(probe_sink, "begin_call", None) is not None:
        checks.append(("begin_call_custody", lambda r: _check_begin_call_custody(sink(), source(), r)))

    try:
        for name, check in checks:
            try:
                await check(f"{rollout_id}-{name.replace('_', '-')}")
            except ConformanceError:
                raise
            except Exception as error:  # noqa: BLE001 - a raising backend is a conformance failure, not a crash.
                raise ConformanceError(name, f"unexpected {type(error).__name__}: {error}") from error
            passed.append(name)
    finally:
        for closable in closables:
            await closable.close()
    return passed


async def _check_put_then_freeze(sink: TokenSink, src: TokenSource, rollout_id: str) -> None:
    name = "put_then_freeze_visibility"
    entry = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[13, 14], request_items=_REQUEST, text="a")
    await sink.put(entry)
    snapshot = await src.freeze(rollout_id)
    _require(len(snapshot.entries) == 1, name, f"expected one entry, got {len(snapshot.entries)}")
    _require(not snapshot.incomplete, name, "clean rollout froze incomplete")
    _require(bool(snapshot.snapshot_id), name, "snapshot_id is empty")
    frozen = snapshot.entries[0]
    _require(
        frozen.model_dump(mode="json") == entry.model_dump(mode="json"),
        name,
        "frozen entry does not round-trip the stored payload",
    )


async def _check_idempotent_reput(sink: TokenSink, src: TokenSource, rollout_id: str) -> None:
    name = "idempotent_reput"
    entry = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[13, 14], request_items=_REQUEST, text="a")
    await sink.put(entry)
    await sink.put(_identical_retry(entry))
    snapshot = await src.freeze(rollout_id)
    _require(len(snapshot.entries) == 1, name, f"byte-identical retry duplicated: {len(snapshot.entries)} entries")
    _require(not snapshot.incomplete, name, "byte-identical retry marked the rollout incomplete")


async def _check_conflicting_reput(sink: TokenSink, src: TokenSource, rollout_id: str) -> None:
    name = "conflicting_reput"
    entry = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[13, 14], request_items=_REQUEST, text="a")
    conflict = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[15], request_items=_REQUEST, text="b")
    await sink.put(entry)
    try:
        await sink.put(conflict)
    except Exception:
        return  # Fail-closed at the writer.
    # Fail-closed at the reader: the rollout must not look trainable.
    snapshot = await src.freeze(rollout_id)
    _require(
        snapshot.incomplete,
        name,
        "conflicting payload for one call id was accepted without raising or marking incomplete",
    )


async def _check_mark_incomplete(sink: TokenSink, source_factory: Callable[[], TokenSource], rollout_id: str) -> None:
    name = "mark_incomplete_durability"
    entry = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[13, 14], request_items=_REQUEST, text="a")
    await sink.put(entry)
    before = await source_factory().freeze(rollout_id)
    # After freeze: the marker must still land and must move the version.
    await sink.mark_incomplete(rollout_id, "call-2")
    after = await source_factory().freeze(rollout_id)
    _require(after.incomplete, name, "a fresh source instance does not see the incomplete marker")
    _require(after.version != before.version, name, "mark_incomplete did not change the observable version")
    retired = await source_factory().drop(rollout_id, snapshot_id=before.snapshot_id, version=before.version)
    _require(not retired, name, "a retirement staled by mark_incomplete succeeded")


async def _check_freeze_idempotency(
    sink: TokenSink, source_factory: Callable[[], TokenSource], rollout_id: str
) -> None:
    name = "freeze_idempotency"
    entry = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[13, 14], request_items=_REQUEST, text="a")
    await sink.put(entry)
    first = await source_factory().freeze(rollout_id)
    second = await source_factory().freeze(rollout_id)
    _require(first.snapshot_id == second.snapshot_id, name, "snapshot_id changed across idempotent freezes")
    _require(first.version == second.version, name, "version changed across idempotent freezes")
    _require(
        {e.model_call_id: e.digest for e in first.entries} == {e.model_call_id: e.digest for e in second.entries},
        name,
        "entry set changed across idempotent freezes",
    )


async def _check_post_freeze_write(sink: TokenSink, src: TokenSource, rollout_id: str) -> None:
    name = "post_freeze_write_safety"
    entry = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[13, 14], request_items=_REQUEST, text="a")
    await sink.put(entry)
    snapshot = await src.freeze(rollout_id)
    late = _make_entry(
        rollout_id, "call-2", prompt=[11, 12, 13, 14], generation=[15], request_items=_REQUEST, text="b"
    )
    try:
        await sink.put(late)
    except Exception:
        return  # A hard fence is acceptable.
    # A relaxed fence must invalidate the consumed snapshot instead.
    retired = await src.drop(rollout_id, snapshot_id=snapshot.snapshot_id, version=snapshot.version)
    _require(not retired, name, "a write raced freeze, yet the stale snapshot was retired")


async def _check_conditional_retirement(sink: TokenSink, src: TokenSource, rollout_id: str) -> None:
    name = "conditional_retirement"
    entry = _make_entry(rollout_id, "call-1", prompt=[11, 12], generation=[13, 14], request_items=_REQUEST, text="a")
    await sink.put(entry)
    snapshot = await src.freeze(rollout_id)
    stale = await src.drop(rollout_id, snapshot_id=snapshot.snapshot_id, version=snapshot.version + 1)
    _require(not stale, name, "a stale-version retirement succeeded")
    evidence = await src.freeze(rollout_id)
    _require(len(evidence.entries) == 1, name, "a failed retirement discarded the evidence")
    retired = await src.drop(rollout_id, snapshot_id=snapshot.snapshot_id, version=snapshot.version)
    _require(retired, name, "retiring the exact consumed snapshot failed")


async def _check_lineage_visibility(
    sink: TokenSink,
    lineage_factory: Callable[[], LineageStore],
    rollout_id: str,
    *,
    fresh_client: bool,
) -> None:
    name = "fresh_client_lineage_visibility" if fresh_client else "lineage_visibility"
    # The non-fresh variant resolves through a store that existed before the write.
    store = lineage_factory() if not fresh_client else None
    entry = _make_entry(
        rollout_id,
        "call-1",
        prompt=[101, 102, 103],
        generation=[201, 202],
        request_items=_REQUEST,
        text="It is sunny.",
    )
    await sink.put(entry)
    # The fresh variant constructs its resolver only after the entry is durable.
    if store is None:
        store = lineage_factory()
    faithful = list(_REQUEST) + list(entry.output_items) + [{"role": "user", "content": "And tomorrow?"}]
    resolution = await store.resolve(rollout_id, faithful)
    _require(
        resolution.status == ParentResolutionStatus.RESOLVED,
        name,
        f"a faithful continuation resolved as {resolution.status}: {resolution.reason}",
    )
    assert resolution.match is not None
    _require(resolution.match.model_call_id == "call-1", name, "resolved to the wrong parent call")
    _require(
        list(resolution.match.cumulative_token_ids) == cumulative_tokens(entry),
        name,
        "resolved parent carries the wrong cumulative tokens",
    )
    rewritten = [{"role": "user", "content": "REWRITTEN"}] + list(entry.output_items)
    rewritten_resolution = await store.resolve(rollout_id, rewritten)
    _require(
        rewritten_resolution.status == ParentResolutionStatus.UNRESOLVED,
        name,
        f"a rewritten context resolved as {rewritten_resolution.status}",
    )
    root = await store.resolve(rollout_id, [{"role": "user", "content": "fresh question"}])
    _require(
        root.status == ParentResolutionStatus.ROOT,
        name,
        f"a request without model history resolved as {root.status}",
    )


async def _check_begin_call_custody(sink: TokenSink, src: TokenSource, rollout_id: str) -> None:
    name = "begin_call_custody"
    await sink.begin_call(rollout_id, "call-lost")  # type: ignore[attr-defined]
    snapshot = await src.freeze(rollout_id)
    _require(snapshot.incomplete, name, "a dangling pre-dispatch intent did not mask the rollout")
