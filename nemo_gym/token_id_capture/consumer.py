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
from nemo_gym.token_id_capture.protocols import TokenCaptureSnapshot, TokenSource
from nemo_gym.token_id_capture.records import RefusalRecord, TokenEntry
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

    rollout_ids = [rollout_id for record in records if (rollout_id := maybe_rollout_id_from_run_body(record))]
    for directory in token_capture_dirs:
        TokenCaptureStore(directory).delete_now(rollout_ids)


def _assemble(
    rollout_id: str,
    entries: list[TokenEntry],
    builder: str,
    model: str,
    *,
    verified_response: dict | None = None,
    explicit_terminal_call_id: str | None = None,
    declared_response_id: str | None = None,
    refusals: tuple[RefusalRecord, ...] = (),
) -> dict:
    # Attribute the verified terminal before building.
    # ``verified_response`` is the scored response from the /run result.
    # Attribution anchors chain selection; its absence falls back to the
    # strict single-chain policy. It must never fail a build.
    try:
        attribution = resolve_terminal(
            entries, verified_response, explicit_terminal_call_id, declared_response_id=declared_response_id
        )
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
        response = project_main_chain_response(rollout_id, out, model=model, refusals=refusals)
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
        "parent_link_failures": dict(notes.parent_link_failures),
        "unresolved_parent_calls": len(notes.unresolved_parent_calls),
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
    elif attribution.attributed or declared_response_id:
        # A declared terminal must be attributable, and an attributed terminal
        # must have a buildable chain. Neither can use the single-chain fallback.
        mask = True
    else:
        # No attribution: the strict single-chain policy applies.
        # A retry of the final call can leave two plausible generations.
        # Mask the rollout when the client-selected generation is unknown.
        mask = bool(unresolved) or bool(notes.unresolved_parent_calls) or notes.roots != 1 or notes.chains != 1
    # An empty delivery must never be trainable, whatever produced it.
    mask = mask or not any(item.get("generation_token_ids") for item in response.get("output", []))
    return {
        "rollout_id": rollout_id,
        "builder": builder,
        "rebuilt_response": response,
        "metrics": metrics,
        "mask_sample": mask,
        "unresolved_retries": list(unresolved),
        "unresolved_parent_calls": list(notes.unresolved_parent_calls),
    }


def _incomplete_masks(built: dict, *, always: bool) -> bool:
    """Whether an incomplete snapshot masks this build.

    Incomplete means some model call registered its capture intent and never
    committed a record (a harness killed at its timeout backstop leaves exactly
    this signature). With ``always`` the build is masked, because that call may
    be the trajectory's real terminal.

    Without ``always`` a delivered terminal attribution keeps the build: the
    attributed terminal is the verified response's final model output, the
    delivered chain reaches it whole, and a call with no record cannot sit
    inside that chain, so the uncaptured call is outside the scored trajectory.
    Any other attribution outcome still masks.
    """
    if always:
        return True
    attribution = (built.get("metrics") or {}).get("terminal_attribution") or {}
    return attribution.get("chain") != "delivered"


def _copy_refusals(built: dict, snapshot: TokenCaptureSnapshot) -> None:
    """Put the snapshot's refusal records on the build's metrics.

    The records travel on every build, masked or failed included, because aggregate reporting
    needs the reason a rollout ended even when it is not trained on. They are also the only way
    to tell a refusal from an output-budget cut: both leave the rebuilt response ``incomplete``
    with reason ``max_output_tokens``. The key is absent when no call was refused.
    """
    if not snapshot.refusals:
        return
    built.setdefault("metrics", {})["refusals"] = [record.model_dump(mode="json") for record in snapshot.refusals]


def mask_incomplete_when_attributed_from_config(global_config_dict) -> bool:
    """Return the ``token_id_capture.mask_incomplete_when_attributed`` setting.

    A caller that holds the Gym global config reads the setting here and passes
    it to the build functions below, which default to the strict behavior.
    """
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    return TokenIdCaptureConfig.model_validate(global_config_dict).token_id_capture.mask_incomplete_when_attributed


def trajectories_for_rollout(
    rollout_id: str,
    token_capture_dirs: list[Path],
    *,
    builder: str = "prefix_merging",
    model: str = "",
    verified_response: dict | None = None,
    explicit_terminal_call_id: str | None = None,
    mask_incomplete_when_attributed: bool = True,
) -> dict | None:
    """Build trajectories from a frozen local token-store snapshot.

    Return ``None`` only when no capture directory is configured.
    Missing records are unsafe and return a masked result.
    An incomplete snapshot masks the result. With
    ``mask_incomplete_when_attributed`` set to ``False`` it masks only when
    terminal attribution did not deliver its chain, because a delivered chain
    places the uncaptured call off the scored path.
    A refused call is reported through ``metrics["refusals"]``; see ``_copy_refusals``.
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
                refusals=snapshot.refusals,
            )
        _copy_refusals(built, snapshot)
        if snapshot.incomplete:
            built.setdefault("metrics", {})["capture_incomplete"] = True
            if _incomplete_masks(built, always=mask_incomplete_when_attributed):
                built["mask_sample"] = True
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
    mask_incomplete_when_attributed: bool = True,
    declared_response_id: str | None = None,
) -> dict | None:
    """Build trajectories from a frozen ``TokenSource`` snapshot.

    ``declared_response_id`` is the served response id the harness reports;
    a declared id that matches no entry masks the rollout.

    Missing records are unsafe and return a masked result.
    An incomplete snapshot masks the result. With
    ``mask_incomplete_when_attributed`` set to ``False`` it masks only when
    terminal attribution did not deliver its chain, because a delivered chain
    places the uncaptured call off the scored path.
    A refused call is reported through ``metrics["refusals"]``; see ``_copy_refusals``.
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
        refusals = snapshot.refusals
        built = await asyncio.to_thread(
            lambda: _assemble(
                rollout_id,
                entries,
                builder,
                model,
                verified_response=verified_response,
                explicit_terminal_call_id=explicit_terminal_call_id,
                declared_response_id=declared_response_id,
                refusals=refusals,
            )
        )
    _copy_refusals(built, snapshot)
    if snapshot.incomplete:
        built.setdefault("metrics", {})["capture_incomplete"] = True
        if _incomplete_masks(built, always=mask_incomplete_when_attributed):
            built["mask_sample"] = True
    built["_capture_snapshot"] = {
        "snapshot_id": snapshot.snapshot_id,
        "version": snapshot.version,
    }
    return built
