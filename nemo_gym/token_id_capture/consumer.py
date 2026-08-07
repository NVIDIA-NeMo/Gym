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

"""The consumer that turns a rollout's captured tokens into trajectories.

This is the single primitive both consumers call after a rollout finishes: Gym's
rollout collection (co-located, reading the token store's files) and a trainer's
finalizer, which passes whatever ``TokenSource`` its transport provides.
The only difference between them is where the ``TokenEntry`` records come from;
the build and projection are identical.

It is deliberately free of any rollout-record or model-server imports, so it
does not couple to those layers. The caller supplies the ``rollout_id`` (Gym's
rollout collection derives it from the record's task/rollout/attempt indices)
and the metrics that describe what the build kept.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nemo_gym.token_id_capture.builder import (
    assert_prefix_contiguity,
    project_main_chain_response,
    run_builder,
)
from nemo_gym.token_id_capture.protocols import TokenSource
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore


logger = logging.getLogger(__name__)


def token_id_capture_dirs_from_config(global_config_dict) -> list[Path]:
    """Resolve the token store directory when training-token capture is enabled, else []."""
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    config = TokenIdCaptureConfig.model_validate(global_config_dict)
    directory = config.resolved_dir()
    # A configured sink replaces the file store, so there is no directory to read back from.
    if not config.enabled or config.token_id_capture.sink is not None:
        return []
    return [directory] if directory is not None else []


def clear_token_captures_for_rollouts(records: list, token_capture_dirs: list[Path]) -> None:
    """Remove stale token records for rollouts about to be dispatched.

    Rollout ids are deterministic and ``TokenCaptureStore.append`` opens in "ab"
    mode, so a rerun that reuses an id would append onto the previous attempt's
    records and the builder would stitch two attempts into one trajectory. The
    caller passes only the rows being dispatched, after any retry suffix has been
    assigned.
    """
    if not token_capture_dirs:
        return
    from nemo_gym.base_responses_api_model import maybe_rollout_id_from_run_body

    for directory in token_capture_dirs:
        store = TokenCaptureStore(directory)
        for record in records:
            rollout_id = maybe_rollout_id_from_run_body(record)
            if rollout_id:
                store.delete(rollout_id)


def _assemble(
    rollout_id: str,
    entries: list[TokenEntry],
    builder: str,
    model: str,
) -> dict:
    # A malformed capture must degrade this one rollout, not take down the caller.
    # Both the contiguity assertion and the flattener raise, and the callers are a
    # rollout-collection loop and a trainer's step loop, where an escaping exception
    # kills a whole batch rather than dropping one sample.
    try:
        out = run_builder(entries, builder)
        for chain in out.chains:
            chain.validate()
        response = project_main_chain_response(rollout_id, out, model=model)
        assert_prefix_contiguity(response)
    except (AssertionError, ValueError, KeyError, IndexError, TypeError) as error:
        logger.warning(
            "Could not build a trajectory for rollout %s from %d captured call(s): %s",
            rollout_id,
            len(entries),
            error,
        )
        return {
            "rollout_id": rollout_id,
            "builder": builder,
            "rebuilt_response": None,
            "mask_sample": True,
            "error": f"{type(error).__name__}: {error}",
            "metrics": {"n_calls": len(entries)},
        }

    notes = out.notes
    # Surface what the build dropped, so a rollout that trained on one of five calls does not look
    # like one that trained on all five.
    metrics = {
        "n_calls": len(entries),
        "chains": notes.chains,
        "quarantined_calls": len(out.quarantined),
        "quarantined_fraction": round(len(out.quarantined) / len(entries), 4) if entries else 0.0,
        "delivered_fraction": notes.delivered_fraction,
        "generated_tokens_captured": notes.generated_tokens_captured,
        "generated_tokens_delivered": notes.generated_tokens_delivered,
        "parent_link_fallbacks": dict(notes.parent_link_fallbacks),
        # Calls the model returned with no generated tokens. They carry no training signal and are
        # kept out of the chain; a non-zero count usually means the output budget or a content
        # filter is cutting generations off.
        "empty_generation_calls": len(notes.empty_generation_calls),
    }
    unresolved = notes.unresolved_retries
    return {
        "rollout_id": rollout_id,
        "builder": builder,
        "rebuilt_response": response,
        "metrics": metrics,
        # A retry of the final call leaves two generations with no way to tell which one the client
        # received. Training on the wrong one is silently off-policy, so the rollout is masked.
        "mask_sample": bool(unresolved),
        "unresolved_retries": list(unresolved),
    }


def trajectories_for_rollout(
    rollout_id: str,
    token_capture_dirs: list[Path],
    *,
    builder: str = "prefix_merging",
    model: str = "",
) -> dict | None:
    """Co-located path: read the rollout's tokens from the store files and build its trajectories.

    come from the verifier result and ride the trajectory; they are not read from the token store.
    Returns ``None`` when no tokens were captured for the rollout (capture off, or a dialect the
    engine returned no ids for). Mirrors how evaluation capture is merged into a rollout record.
    """
    for directory in token_capture_dirs:
        store = TokenCaptureStore(directory)
        entries = store.read_entries(rollout_id)
        if entries:
            built = _assemble(rollout_id, entries, builder, model)
            if store.is_incomplete(rollout_id):
                # At least one call of this rollout failed to capture. The chain we built may look
                # perfectly contiguous while being missing a turn, so mask rather than train on it.
                built["mask_sample"] = True
                built.setdefault("metrics", {})["capture_incomplete"] = True
            return built
    return None


async def trajectories_from_source(
    rollout_id: str,
    source: TokenSource,
    *,
    builder: str = "prefix_merging",
    model: str = "",
) -> dict | None:
    """Read the rollout's records through a ``TokenSource`` and build its trajectories.

    Returns ``None`` when nothing was recorded for it.
    """
    entries = await source.tokens_for(rollout_id)
    if not entries:
        return None
    built = _assemble(rollout_id, entries, builder, model)
    # A rollout that lost a call can stitch into a chain that looks perfectly contiguous while
    # missing a turn, so this is the only way to tell. A transport that cannot report it says so
    # by returning False.
    if source.is_incomplete(rollout_id):
        built["mask_sample"] = True
        built.setdefault("metrics", {})["capture_incomplete"] = True
    return built
