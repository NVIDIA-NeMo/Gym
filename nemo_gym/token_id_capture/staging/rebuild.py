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

"""Pure rebuild of training rows from verified staging snapshots.

Two functions every framework's finalizer calls after it has fetched and
digest-verified a rollout's staged rows:

* ``snapshots_to_entries`` -- the hash-free, mask-driven inverse of the
  worker's ``build_staging_delta``: walk rows in admission order, reconstruct
  each call's full prompt and generation from its delta and its parent's
  cumulative sequence.
* ``linearize`` -- turn the per-call deltas into training rows under a
  policy. The MVP policy is ``main_chain_only``: walk the manifest's parent
  pointers from ``terminal_hint`` back to a root and concatenate that chain's
  deltas into one row (ids, loss mask, logprobs). Calls off the main chain
  are rebuilt (so they are *verified*) but not trained.

The semantics here are the training-row contract: identical for every
framework, exercised by the golden fixtures in ``conformance/``.

This module is part of the dependency-free capture core: stdlib + pydantic
(via ``records``) only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from nemo_gym.token_id_capture.staging.records import CallRecord, StagedCallSnapshot


class RebuildError(ValueError):
    """A snapshot set that could not have been produced by the worker's
    delta builder (or a manifest that does not describe it)."""


@dataclass(frozen=True)
class RebuiltEntry:
    """One model call reconstructed from its staged delta: the exact prompt
    the engine ran on and what it generated. ``seq`` is the admission index."""

    rollout_id: str
    call_id: str
    seq: int
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    weight_version: Optional[int] = None


@dataclass(frozen=True)
class LinearizedRow:
    """One training row: the flattened main chain of a rollout.

    ``token_mask`` is 1.0 exactly on policy-sampled tokens; ``logprobs`` is
    0.0 off-mask. ``call_ids`` lists the chain's calls root-first, and
    ``prompt_len`` is the length of the root call's carried prompt (the
    row's untrained prefix)."""

    rollout_id: str
    token_ids: list[int]
    token_mask: list[float]
    logprobs: list[float]
    call_ids: list[str]
    prompt_len: int
    weight_versions: list[int] = field(default_factory=list)


def _carry_boundary(snapshot: StagedCallSnapshot) -> int:
    """The index where a delta's 0.0 prompt-carry prefix ends. Any 0.0 after
    a 1.0 means the delta was not produced by ``build_staging_delta``."""
    boundary = 0
    for mask_value in snapshot.token_mask_delta:
        if mask_value == 0.0:
            boundary += 1
        else:
            break
    if any(mask_value == 0.0 for mask_value in snapshot.token_mask_delta[boundary:]):
        raise RebuildError(
            f"call {snapshot.call_id}: token_mask_delta is not a prompt-carry prefix followed by generated tokens"
        )
    return boundary


def snapshots_to_entries(rollout_id: str, snapshots: list[StagedCallSnapshot]) -> list[RebuiltEntry]:
    """Rebuild per-call entries from staged snapshots in admission order.

    Walking rows in order, each row's full prompt is its parent's cumulative
    sequence truncated to ``prev_len`` plus the row's prompt-carry tokens
    (mask 0.0 prefix); its generation is the mask 1.0 suffix. A parentless
    row is self-contained (``prev_len == 0``). This is the exact inverse of
    the worker's ``build_staging_delta``.
    """
    entries: list[RebuiltEntry] = []
    cumulative_by_call: dict[str, list[int]] = {}
    for seq, snapshot in enumerate(snapshots):
        if snapshot.parent_call_id is not None:
            base = cumulative_by_call.get(snapshot.parent_call_id)
            if base is None:
                raise RebuildError(
                    f"call {snapshot.call_id}: parent {snapshot.parent_call_id} precedes it in no snapshot"
                )
            if len(base) != snapshot.prev_len:
                raise RebuildError(
                    f"call {snapshot.call_id}: prev_len={snapshot.prev_len} does not equal parent length {len(base)}"
                )
        else:
            base = []
            if snapshot.prev_len != 0:
                raise RebuildError(
                    f"call {snapshot.call_id}: parentless row must be self-contained, got prev_len={snapshot.prev_len}"
                )
        if not (len(snapshot.token_ids_delta) == len(snapshot.token_mask_delta) == len(snapshot.logprobs_delta)):
            raise RebuildError(f"call {snapshot.call_id}: misaligned delta arrays")
        boundary = _carry_boundary(snapshot)
        prompt_token_ids = base[: snapshot.prev_len] + snapshot.token_ids_delta[:boundary]
        generation_token_ids = snapshot.token_ids_delta[boundary:]
        generation_log_probs = snapshot.logprobs_delta[boundary:]
        entries.append(
            RebuiltEntry(
                rollout_id=rollout_id,
                call_id=snapshot.call_id,
                seq=seq,
                prompt_token_ids=prompt_token_ids,
                generation_token_ids=generation_token_ids,
                generation_log_probs=generation_log_probs,
                weight_version=snapshot.weight_version,
            )
        )
        cumulative_by_call[snapshot.call_id] = prompt_token_ids + generation_token_ids
    return entries


LinearizePolicy = Literal["main_chain_only"]


def main_chain_call_ids(manifest: list[CallRecord], terminal_hint: Optional[str]) -> list[str]:
    """The manifest calls on the terminal call's root-to-terminal chain.

    ``terminal_hint`` defaults to the manifest's last call. Parent pointers
    are explicit, so this is a dictionary walk, not a content match; a parent
    missing from the manifest (failed or never committed) is a contract
    violation."""
    if not manifest:
        return []
    by_id = {record.call_id: record for record in manifest}
    terminal = terminal_hint if terminal_hint is not None else manifest[-1].call_id
    if terminal not in by_id:
        raise RebuildError(f"terminal call {terminal} is not in the manifest")
    chain: list[str] = []
    cursor: Optional[str] = terminal
    while cursor is not None:
        record = by_id.get(cursor)
        if record is None:
            raise RebuildError(f"chain parent {cursor} is not in the manifest")
        chain.append(cursor)
        if len(chain) > len(manifest):
            raise RebuildError("manifest parent pointers form a cycle")
        cursor = record.parent_call_id
    chain.reverse()
    return chain


def linearize(
    rollout_id: str,
    snapshots: list[StagedCallSnapshot],
    manifest: list[CallRecord],
    *,
    policy: LinearizePolicy = "main_chain_only",
    terminal_hint: Optional[str] = None,
) -> LinearizedRow:
    """Flatten the main chain's deltas into one training row.

    ``snapshots`` must be the manifest's rows in manifest (admission) order;
    they are fully rebuilt first, so every row -- on or off the chain -- is
    shape-verified. The returned row is the concatenation of the chain's
    deltas: token ids as staged, mask 1.0 exactly on generated tokens,
    logprobs 0.0 off-mask.
    """
    if policy != "main_chain_only":
        raise NotImplementedError(f"linearize policy {policy!r} (MVP supports main_chain_only)")
    if len(snapshots) != len(manifest) or any(
        snapshot.call_id != record.call_id for snapshot, record in zip(snapshots, manifest)
    ):
        raise RebuildError("snapshots do not match the manifest rows in order")
    # Full-forest rebuild: verifies every row's shape and parent chaining.
    snapshots_to_entries(rollout_id, snapshots)

    chain = main_chain_call_ids(manifest, terminal_hint)
    by_id = {snapshot.call_id: snapshot for snapshot in snapshots}
    token_ids: list[int] = []
    token_mask: list[float] = []
    logprobs: list[float] = []
    weight_versions: list[int] = []
    prompt_len: Optional[int] = None
    expected_prev_len = 0
    for call_id in chain:
        snapshot = by_id[call_id]
        if snapshot.prev_len != expected_prev_len:
            raise RebuildError(
                f"call {call_id}: chain prev_len={snapshot.prev_len} does not equal "
                f"accumulated length {expected_prev_len}"
            )
        if prompt_len is None:
            prompt_len = _carry_boundary(snapshot)
        token_ids.extend(snapshot.token_ids_delta)
        token_mask.extend(snapshot.token_mask_delta)
        logprobs.extend(snapshot.logprobs_delta)
        if snapshot.weight_version is not None:
            weight_versions.append(snapshot.weight_version)
        expected_prev_len += len(snapshot.token_ids_delta)
    return LinearizedRow(
        rollout_id=rollout_id,
        token_ids=token_ids,
        token_mask=token_mask,
        logprobs=logprobs,
        call_ids=chain,
        prompt_len=prompt_len or 0,
        weight_versions=weight_versions,
    )
