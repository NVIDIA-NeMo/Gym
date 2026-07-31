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

"""Conformance kit: replay golden call sequences and assert byte-exact results.

A fixture describes one rollout as the worker sees it: an ordered list of
model calls, each with the prompt ids the engine ran on, the generated ids
and logprobs, its parent, and its capture mode. From that, the kit derives --
through the real ``digest``/``lineage``/``rebuild`` code paths -- the staged
records, coords, receipt manifest, and linearized training row, and compares
them against the fixture's frozen expectations.

Two entrypoints:

* ``run_lineage_conformance(fixture)`` -- drives ``RolloutLineage`` directly
  (no storage): admission/commit ordering, manifest, linearized row.
* ``run_sink_source_conformance(fixture, sink, source)`` -- additionally
  round-trips every staged record through a framework's ``StagingSink`` and
  ``StagingSource`` and requires the fetched snapshots to reproduce the same
  digests and the same linearized row byte-for-byte. This is the test every
  framework implementation runs in its CI.

Logprob/mask floats are float32-quantized at the fixture boundary (staging
columns are float32); expectations are stored quantized, so a conforming
implementation must preserve float32 bit patterns exactly.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Optional

from nemo_gym.token_id_capture.staging.digest import build_staging_delta, compute_staging_digest
from nemo_gym.token_id_capture.staging.lineage import RolloutLineage
from nemo_gym.token_id_capture.staging.rebuild import LinearizedRow, linearize, snapshots_to_entries
from nemo_gym.token_id_capture.staging.records import (
    CommitCoords,
    RolloutReceipt,
    StagedCallRecord,
    StagedCallSnapshot,
    staging_key,
)


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


class ConformanceFailure(AssertionError):
    """A framework implementation diverged from the golden expectations."""


def f32(value: float) -> float:
    """Quantize one float to its float32 value (the staging column dtype)."""
    return struct.unpack(">f", struct.pack(">f", value))[0]


def fixture_names() -> list[str]:
    return sorted(path.stem for path in _FIXTURE_DIR.glob("*.json"))


def load_fixture(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / f"{name}.json").open("rb") as handle:
        return json.load(handle)


def build_fixture_artifacts(
    fixture: dict[str, Any],
) -> tuple[list[StagedCallRecord], list[CommitCoords], RolloutReceipt, Optional[LinearizedRow]]:
    """Derive records, coords, receipt, and the linearized row from a fixture's
    call sequence via the real digest/lineage/rebuild code paths."""
    rollout_id = fixture["rollout_id"]
    lineage = RolloutLineage(rollout_id, now=0.0)
    records: list[StagedCallRecord] = []
    coords_list: list[CommitCoords] = []
    cum_len_by_call: dict[str, int] = {}
    for call in fixture["calls"]:
        call_id = call["call_id"]
        parent = call.get("parent_call_id")
        mode = call.get("mode", "token_in" if parent else "text")
        admitted = lineage.admit(call_id, parent_call_id=parent, mode=mode, now=0.0)
        if call.get("disposition") == "capture_failed":
            # The stage failed at the worker: no durable row, poison coords only.
            lineage.commit(
                CommitCoords(
                    rollout_id=rollout_id,
                    call_id=call_id,
                    parent_call_id=parent,
                    delta_len=0,
                    cum_len=admitted.prev_len,
                    digest="",
                    staging_key=staging_key(rollout_id, call_id),
                    weight_version=int(call["weight_version"]),
                    disposition="capture_failed",
                ),
                now=0.0,
            )
            continue
        prompt_ids = [int(t) for t in call["prompt_token_ids"]]
        generated = [int(t) for t in call["generation_token_ids"]]
        logprobs = [f32(p) for p in call["generation_log_probs"]]
        ids_delta, mask_delta, lp_delta = build_staging_delta(
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated,
            generated_logprobs=logprobs,
            prev_len=admitted.prev_len,
        )
        digest = compute_staging_digest(
            rollout_id=rollout_id,
            call_id=call_id,
            prev_len=admitted.prev_len,
            token_ids_delta=ids_delta,
            token_mask_delta=mask_delta,
            logprobs_delta=lp_delta,
        )
        weight_version = int(call["weight_version"])
        record = StagedCallRecord(
            rollout_id=rollout_id,
            call_id=call_id,
            parent_call_id=parent,
            prev_len=admitted.prev_len,
            new_len=admitted.prev_len + len(ids_delta),
            weight_version=weight_version,
            digest=digest,
            token_ids_delta=ids_delta,
            token_mask_delta=mask_delta,
            generation_logprobs_delta=lp_delta,
        )
        coords = CommitCoords(
            rollout_id=rollout_id,
            call_id=call_id,
            parent_call_id=parent,
            delta_len=len(ids_delta),
            cum_len=record.new_len,
            digest=digest,
            staging_key=record.staging_key,
            weight_version=weight_version,
            token_ids_delta=ids_delta,
        )
        records.append(record)
        coords_list.append(coords)
        lineage.commit(coords, now=0.0)
        cum_len_by_call[call_id] = record.new_len
    receipt = lineage.seal(
        reward=fixture.get("reward"),
        terminal_call_id=fixture.get("terminal_call_id"),
        now=0.0,
    )
    row: Optional[LinearizedRow] = None
    if receipt.manifest:
        snapshots = [_record_to_snapshot(record) for record in records]
        row = linearize(
            rollout_id,
            snapshots,
            receipt.manifest,
            terminal_hint=receipt.terminal_call_id,
        )
    return records, coords_list, receipt, row


def _record_to_snapshot(record: StagedCallRecord) -> StagedCallSnapshot:
    return StagedCallSnapshot(
        call_id=record.call_id,
        prev_len=record.prev_len,
        token_ids_delta=list(record.token_ids_delta),
        token_mask_delta=list(record.token_mask_delta),
        logprobs_delta=list(record.generation_logprobs_delta),
        weight_version=record.weight_version,
        parent_call_id=record.parent_call_id,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceFailure(message)


def _assert_expected(
    fixture: dict[str, Any], records: list[StagedCallRecord], receipt: RolloutReceipt, row: Optional[LinearizedRow]
) -> None:
    expected = fixture["expected"]
    _require(
        [record.digest for record in records] == expected["digests"],
        "staged digests diverge from golden digests",
    )
    manifest_dump = [record.model_dump() for record in receipt.manifest]
    _require(manifest_dump == expected["manifest"], "receipt manifest diverges from golden manifest")
    _require(receipt.terminal_call_id == expected["terminal_call_id"], "terminal call diverges")
    _require(receipt.capture_poisoned == expected.get("capture_poisoned", False), "poison flag diverges")
    if expected.get("row") is None:
        _require(row is None, "expected no training row")
        return
    _require(row is not None, "expected a training row")
    assert row is not None
    _require(row.token_ids == expected["row"]["token_ids"], "row token ids diverge")
    _require(row.token_mask == [f32(m) for m in expected["row"]["token_mask"]], "row mask diverges")
    _require(row.logprobs == [f32(p) for p in expected["row"]["logprobs"]], "row logprobs diverge")
    _require(row.call_ids == expected["row"]["call_ids"], "row chain diverges")
    _require(row.prompt_len == expected["row"]["prompt_len"], "row prompt_len diverges")


def run_lineage_conformance(fixture: dict[str, Any]) -> None:
    """Golden check with no storage in the loop (gate/direct-drive parity)."""
    records, _, receipt, row = build_fixture_artifacts(fixture)
    _assert_expected(fixture, records, receipt, row)


def run_sink_source_conformance(fixture: dict[str, Any], sink: Any, source: Any) -> None:
    """Golden check through a framework's ``StagingSink``/``StagingSource``.

    Stages every record, fetches it back by staging key, and requires the
    fetched snapshots to recompute the same digests and linearize to the same
    row the direct drive produced.
    """
    records, _, receipt, row = build_fixture_artifacts(fixture)
    _assert_expected(fixture, records, receipt, row)

    rollout_id = fixture["rollout_id"]
    for record in records:
        result = sink.stage(record)
        _require(result.ok, f"sink.stage failed: {result.error}")
        _require(
            result.staging_key == staging_key(rollout_id, record.call_id),
            "sink returned a mismatched staging key",
        )
    keys = [record.staging_key for record in records]
    snapshots = source.fetch(keys)
    _require(len(snapshots) == len(records), "source returned wrong row count")
    for record, snapshot in zip(records, snapshots):
        _require(snapshot.call_id == record.call_id, "fetched snapshot order diverges")
        recomputed = compute_staging_digest(
            rollout_id=rollout_id,
            call_id=snapshot.call_id,
            prev_len=snapshot.prev_len,
            token_ids_delta=snapshot.token_ids_delta,
            token_mask_delta=snapshot.token_mask_delta,
            logprobs_delta=snapshot.logprobs_delta,
        )
        _require(
            recomputed == record.digest,
            f"call {snapshot.call_id}: storage round-trip changed the digest (dtype drift or corruption)",
        )
    # Parent pointers are lineage state, not storage state: rejoin them from
    # the manifest before rebuilding, as the finalizer does.
    parent_by_call = {rec.call_id: rec.parent_call_id for rec in receipt.manifest}
    rejoined = [
        snapshot.model_copy(update={"parent_call_id": parent_by_call.get(snapshot.call_id)}) for snapshot in snapshots
    ]
    snapshots_to_entries(rollout_id, rejoined)
    if receipt.manifest:
        fetched_row = linearize(rollout_id, rejoined, receipt.manifest, terminal_hint=receipt.terminal_call_id)
        _require(fetched_row == row, "row rebuilt from fetched snapshots diverges")


def regenerate_expectations(fixture: dict[str, Any]) -> dict[str, Any]:
    """Return the fixture with freshly derived expectations (maintainer tool;
    used once at S1 to freeze the goldens)."""
    records, _, receipt, row = build_fixture_artifacts(fixture)
    out = dict(fixture)
    out["expected"] = {
        "digests": [record.digest for record in records],
        "manifest": [record.model_dump() for record in receipt.manifest],
        "terminal_call_id": receipt.terminal_call_id,
        "capture_poisoned": receipt.capture_poisoned,
        "row": None
        if row is None
        else {
            "token_ids": row.token_ids,
            "token_mask": row.token_mask,
            "logprobs": row.logprobs,
            "call_ids": row.call_ids,
            "prompt_len": row.prompt_len,
        },
    }
    return out
