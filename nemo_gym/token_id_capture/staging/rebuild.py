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

"""The finalizer's rebuild: staged deltas -> one training row.

Thin, terminal-aware layer over the base trajectory builder. The heavy
lifting -- parent resolution (verified links with digest checks), retry
sibling handling, quarantine, chain construction -- is the base's
``run_builder(prefix_merging)`` (``builder.py``). This module contributes
exactly what training custody needs on top of it:

* ``snapshots_to_entries`` -- the exact inverse of the worker's
  ``build_staging_delta``: staged deltas back into per-call (prompt,
  generation, logprobs) entries.
* terminal-aware chain selection -- the base picks its main chain by
  generated-token mass, which silently mispicks when a sub-agent fork
  out-generates the main conversation. The gate knows the rollout's terminal
  call at seal time; ``linearize`` selects the chain that ends in it.
* ``LinearizedRow`` assembly -- token/mask/logprob arrays with per-call
  weight versions, the row a trainer ingests.
* fail-loud custody -- an unresolved final-call retry or a terminal call
  that fell out of the rebuilt chains raises ``RebuildError``; the caller
  publishes a placeholder rather than training on a guess.

Part of the dependency-free staging core (stdlib + pydantic via ``records``
and the base builder only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from nemo_gym.token_id_capture.builder import Chain, run_builder
from nemo_gym.token_id_capture.records import TokenEntry
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
    """One training row: the flattened delivered chain of a rollout.

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


def _to_token_entries(entries: list[RebuiltEntry], snapshots: list[StagedCallSnapshot]) -> list[TokenEntry]:
    """Adapt rebuilt entries to the base builder's ``TokenEntry`` shape.

    The recorded parent link rides along so ``prefix_merging`` takes its
    verified O(1) path; ``digest`` is left unset so the builder verifies the
    link against the actual cumulative tokens. ``weight_version`` rides as an
    extra field (``TokenEntry`` allows extras) and comes back out on the row.
    """
    token_entries: list[TokenEntry] = []
    for entry, snapshot in zip(entries, snapshots):
        token_entries.append(
            TokenEntry(
                rollout_id=entry.rollout_id,
                model_call_id=entry.call_id,
                prompt_token_ids=entry.prompt_token_ids,
                generation_token_ids=entry.generation_token_ids,
                generation_log_probs=entry.generation_log_probs,
                parent_call_id=snapshot.parent_call_id,
                cum_len=len(entry.prompt_token_ids) + len(entry.generation_token_ids),
                weight_version=entry.weight_version,
            )
        )
    return token_entries


def _select_chain(chains: list[Chain], terminal_hint: Optional[str]) -> Chain:
    """Terminal-aware chain selection (the ~20 lines that override the base).

    With a terminal hint, the delivered chain is the one whose leaf is the
    sealed terminal call -- generated-token mass is not consulted, so a
    sub-agent fork that out-generates the main conversation cannot win.
    Without a hint, defer to the base's main-chain pick.
    """
    if not chains:
        raise RebuildError("no chains rebuilt (all calls empty or quarantined)")
    if terminal_hint is None:
        for chain in chains:
            if chain.chain_id == "main":
                return chain
        return chains[0]
    for chain in chains:
        if chain.links and chain.links[-1].entry.model_call_id == terminal_hint:
            return chain
    raise RebuildError(f"terminal call {terminal_hint} is not the leaf of any rebuilt chain")


def linearize(
    rollout_id: str,
    snapshots: list[StagedCallSnapshot],
    manifest: list[CallRecord],
    *,
    terminal_hint: Optional[str] = None,
) -> LinearizedRow:
    """Rebuild the delivered chain of a rollout as one training row.

    ``snapshots`` must be in manifest (admission) order -- the finalizer
    fetches them by manifest key, which preserves it. Chain construction is
    delegated to the base ``run_builder(prefix_merging)``; selection is
    terminal-aware (see ``_select_chain``). Raises ``RebuildError`` when the
    row cannot be trusted: unresolved final-call retries on the delivered
    chain, a quarantined or missing terminal call, or snapshots that disagree
    with the manifest.
    """
    if len(snapshots) != len(manifest):
        raise RebuildError(f"{len(snapshots)} snapshots for a manifest of {len(manifest)} calls")
    for snapshot, record in zip(snapshots, manifest):
        if snapshot.call_id != record.call_id:
            raise RebuildError(f"snapshot order diverges from manifest at call {record.call_id}")

    entries = snapshots_to_entries(rollout_id, snapshots)
    output = run_builder(_to_token_entries(entries, snapshots), builder="prefix_merging")
    chain = _select_chain(output.chains, terminal_hint)
    chain.validate()

    delivered_calls = {link.entry.model_call_id for link in chain.links}
    unresolved = delivered_calls & set(output.notes.unresolved_retries)
    if unresolved:
        raise RebuildError(f"unresolved retry on the delivered chain: {sorted(unresolved)}")

    token_ids: list[int] = list(chain.root_prompt)
    token_mask: list[float] = [0.0] * len(chain.root_prompt)
    logprobs: list[float] = [0.0] * len(chain.root_prompt)
    call_ids: list[str] = []
    weight_versions: list[int] = []
    for step, link in enumerate(chain.links):
        interstitial = link.interstitial if step > 0 else []
        token_ids.extend(interstitial)
        token_mask.extend([0.0] * len(interstitial))
        logprobs.extend([0.0] * len(interstitial))
        generated = link.entry.generation_token_ids
        token_ids.extend(generated)
        token_mask.extend([1.0] * len(generated))
        logprobs.extend(link.entry.generation_log_probs)
        call_ids.append(link.entry.model_call_id)
        weight_version = getattr(link.entry, "weight_version", None)
        weight_versions.append(0 if weight_version is None else int(weight_version))

    return LinearizedRow(
        rollout_id=rollout_id,
        token_ids=token_ids,
        token_mask=token_mask,
        logprobs=logprobs,
        call_ids=call_ids,
        prompt_len=len(chain.root_prompt),
        weight_versions=weight_versions,
    )
