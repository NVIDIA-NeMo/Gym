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

"""Turn a rollout's frozen token capture into trajectories.

Gym rollout collection and trainer finalization use this consumer.
Gym reads a frozen snapshot from the local token store.
A trainer freezes the ``TokenSource`` provided by its transport.
Both paths pass snapshot entries through the same build and projection.
Single-response delivery rejects ``per_request`` because it can return multiple trajectories.

This module does not import rollout-record or model-server modules.
The caller supplies the ``rollout_id``.
Gym derives that ID from task, rollout, and attempt indices.
The result includes metrics that describe the build.
"""

from __future__ import annotations

import asyncio
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
from nemo_gym.token_id_capture.terminal import TerminalAttribution, resolve_terminal


logger = logging.getLogger(__name__)


def _failed_build(rollout_id: str, builder: str, error: str, *, n_calls: int = 0) -> dict:
    return {
        "rollout_id": rollout_id,
        "builder": builder,
        "rebuilt_response": None,
        "mask_sample": True,
        "error": error,
        "metrics": {"n_calls": n_calls},
    }


def token_id_capture_dirs_from_config(global_config_dict) -> list[Path]:
    """Return the enabled token store directory or an empty list."""
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    config = TokenIdCaptureConfig.model_validate(global_config_dict)
    directory = config.resolved_dir()
    # A configured sink replaces the readable file store.
    if not config.enabled or config.token_id_capture.sink is not None:
        return []
    return [directory] if directory is not None else []


def clear_token_captures_for_rollouts(records: list, token_capture_dirs: list[Path]) -> None:
    """Remove stale token records for rollouts about to be dispatched.

    Rollout IDs are deterministic.
    ``TokenCaptureStore.append`` uses append mode.
    A reused ID would append records to a previous attempt.
    The builder could then combine two attempts.
    The caller passes only rows ready for dispatch.
    Retry suffixes must already be assigned.
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
    *,
    verified_response: dict | None = None,
    explicit_terminal_call_id: str | None = None,
) -> dict:
    if builder == "per_request":
        # Single-response delivery cannot represent multiple trajectories.
        return _failed_build(
            rollout_id,
            builder,
            "per_request returns multiple trajectories and is not supported by single-response delivery",
            n_calls=len(entries),
        )
    # Attribute the verified terminal before building.
    # ``verified_response`` is the scored response from the /run result.
    # Attribution anchors chain selection; its absence falls back to the
    # strict single-chain policy. It must never fail a build.
    try:
        attribution = resolve_terminal(entries, verified_response, explicit_terminal_call_id)
    except Exception:
        logger.warning("Terminal attribution failed for rollout %s.", rollout_id, exc_info=True)
        attribution = TerminalAttribution(None, reason="attribution_error")

    # Mask a malformed rollout instead of failing the caller.
    # The contiguity check and projection can raise.
    # An uncaught exception could fail a full rollout or training batch.
    def _attribution_metric(chain_status: str) -> dict:
        return {
            "method": attribution.method or "none",
            "call_id": attribution.model_call_id,
            "reason": attribution.reason,
            "chain": chain_status,
        }

    try:
        out = run_builder(entries, builder, terminal_call_id=attribution.model_call_id)
    except (AssertionError, ValueError, KeyError, IndexError, TypeError) as error:
        logger.warning(
            "Could not build a trajectory for rollout %s from %d captured call(s): %s",
            rollout_id,
            len(entries),
            error,
        )
        failed = _failed_build(rollout_id, builder, f"{type(error).__name__}: {error}", n_calls=len(entries))
        failed["metrics"]["terminal_attribution"] = _attribution_metric("error")
        return failed
    try:
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
        failed = _failed_build(rollout_id, builder, f"{type(error).__name__}: {error}", n_calls=len(entries))
        # The builder's verdict survives a failed projection.
        # A broken terminal chain must stay diagnosable in aggregate metrics.
        failed["metrics"]["terminal_attribution"] = _attribution_metric(out.notes.terminal_chain or "error")
        return failed

    notes = out.notes
    # Report everything dropped by the build.
    # A partial build must not appear complete.
    metrics = {
        "n_calls": len(entries),
        # Attribution names the call whose response the verifier scored.
        # ``chain`` reports what the builder did with that name.
        "terminal_attribution": _attribution_metric(notes.terminal_chain),
        "chains": notes.chains,
        "roots": notes.roots,
        "quarantined_calls": len(out.quarantined),
        "quarantined_fraction": round(len(out.quarantined) / len(entries), 4) if entries else 0.0,
        "delivered_fraction": notes.delivered_fraction,
        "generated_tokens_captured": notes.generated_tokens_captured,
        "generated_tokens_delivered": notes.generated_tokens_delivered,
        # Calls without generated tokens have no training signal.
        # A nonzero count can indicate an output-budget or content-filter cutoff.
        "empty_generation_calls": len(notes.empty_generation_calls),
    }
    unresolved = notes.unresolved_retries
    if notes.terminal_chain == "delivered":
        # The verified chain is attributed and intact.
        # Off-path calls (auxiliary calls, sub-agent forks, abandoned retries)
        # are excluded from delivery instead of masking the rollout.
        mask = False
    elif attribution.attributed:
        # Attribution succeeded but its chain is broken or unbuildable.
        # The rollout's verified trajectory is known and undeliverable.
        mask = True
    else:
        # No attribution: the strict single-chain policy applies.
        # A retry of the final call can leave two plausible generations.
        # Mask the rollout when the client-selected generation is unknown.
        mask = bool(unresolved) or notes.roots != 1 or notes.chains != 1
    return {
        "rollout_id": rollout_id,
        "builder": builder,
        "rebuilt_response": response,
        "metrics": metrics,
        "mask_sample": mask,
        "unresolved_retries": list(unresolved),
    }


def trajectories_for_rollout(
    rollout_id: str,
    token_capture_dirs: list[Path],
    *,
    builder: str = "prefix_merging",
    model: str = "",
    verified_response: dict | None = None,
    explicit_terminal_call_id: str | None = None,
) -> dict | None:
    """Build trajectories from a frozen local token-store snapshot.

    Return ``None`` only when no capture directory is configured.
    Missing records are unsafe and return a masked result.
    An incomplete snapshot is unsafe and returns a masked result.
    """
    for directory in token_capture_dirs:
        store = TokenCaptureStore(directory)
        try:
            snapshot = store.freeze_now(rollout_id)
        except Exception as error:
            logger.warning("Could not freeze token capture for rollout %s.", rollout_id, exc_info=True)
            return _failed_build(rollout_id, builder, f"{type(error).__name__}: {error}")
        if not snapshot.entries:
            built = _failed_build(rollout_id, builder, "capture contains no token records")
        else:
            built = _assemble(
                rollout_id,
                list(snapshot.entries),
                builder,
                model,
                verified_response=verified_response,
                explicit_terminal_call_id=explicit_terminal_call_id,
            )
        if snapshot.incomplete:
            built["mask_sample"] = True
            built.setdefault("metrics", {})["capture_incomplete"] = True
        built["_capture_snapshot"] = {
            "snapshot_id": snapshot.snapshot_id,
            "version": snapshot.version,
        }
        return built
    return None


async def trajectories_from_source(
    rollout_id: str,
    source: TokenSource,
    *,
    builder: str = "prefix_merging",
    model: str = "",
    verified_response: dict | None = None,
    explicit_terminal_call_id: str | None = None,
) -> dict | None:
    """Build trajectories from a frozen ``TokenSource`` snapshot.

    Missing records are unsafe and return a masked result.
    An incomplete snapshot is unsafe and returns a masked result.
    """
    try:
        snapshot = await source.freeze(rollout_id)
    except Exception as error:
        logger.warning("Could not freeze token capture for rollout %s.", rollout_id, exc_info=True)
        return _failed_build(rollout_id, builder, f"{type(error).__name__}: {error}")
    if not snapshot.entries:
        built = _failed_build(rollout_id, builder, "capture contains no token records")
    else:
        entries = list(snapshot.entries)
        built = await asyncio.to_thread(
            lambda: _assemble(
                rollout_id,
                entries,
                builder,
                model,
                verified_response=verified_response,
                explicit_terminal_call_id=explicit_terminal_call_id,
            )
        )
    if snapshot.incomplete:
        built["mask_sample"] = True
        built.setdefault("metrics", {})["capture_incomplete"] = True
    built["_capture_snapshot"] = {
        "snapshot_id": snapshot.snapshot_id,
        "version": snapshot.version,
    }
    return built
