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

"""Build a token-bearing record from one finished rollout.

The finalizer freezes the rollout's captured model calls.
It rebuilds ``response.output`` from that frozen snapshot.
It does not retire the snapshot.
The caller may retire it only after durable handoff.
Retirement uses the frozen ``snapshot_id`` and version.

The caller provides both the rollout record and its ``TokenSource``.
Gym resolves its source from Gym configuration.
A training framework may provide a source from its own data plane.
Training correlation must preserve ``/ng-rollout/<id>/training-token-capture``.

Existing token ids are the policy's sampled data.
The finalizer leaves a rollout containing any token ids unchanged.
Failed or masked builds retain their capture evidence.
"""

from __future__ import annotations

import warnings

from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.token_id_capture.consumer import trajectories_from_source
from nemo_gym.token_id_capture.protocols import TokenSource


# Attach token-capture health to each rollout record.
# It reports build losses and masking reasons.
TOKEN_CAPTURE_KEY = "_ng_token_capture"

# A consumer reads this top-level field to exclude a rollout from the loss.
# The field stays outside TOKEN_CAPTURE_KEY for direct access.
MASK_SAMPLE_KEY = "mask_sample"
_REDUNDANT_CAPTURE_KEY = "_redundant_capture"

# A caller that knows the kept model call names it here on the result.
# A gate seal or an agent-declared terminal id uses this key.
# It joins with the response-id and content witnesses for terminal attribution.
TERMINAL_CALL_KEY = "_ng_terminal_model_call_id"


def rollout_carries_token_ids(result: dict) -> bool:
    """Whether this rollout already holds what training needs.

    Return true when any output item carries generated token ids.
    These ids are what the policy sampled.
    They take precedence over a reconstruction that may differ.
    Partial token coverage must remain visible instead of being overwritten.
    """
    response = result.get("response")
    if not isinstance(response, dict):
        return False
    return any(isinstance(item, dict) and item.get("generation_token_ids") for item in (response.get("output") or []))


def _unusable(result: dict, error: str, message: str) -> dict:
    """Mask a rollout that needed token ids and could not get them.

    An unmasked rollout would appear healthy until it reaches the trainer.
    The record retains the reason for aggregate reporting.
    """
    warnings.warn(message, stacklevel=3)
    metrics = {"n_calls": 0, "error": error}
    result[MASK_SAMPLE_KEY] = True
    result[TOKEN_CAPTURE_KEY] = metrics
    return {"rebuilt_response": None, MASK_SAMPLE_KEY: True, "error": error, "metrics": metrics}


async def finalize_rollout_token_capture(result: dict, source: TokenSource | None) -> dict | None:
    """Rebuild one finished rollout record's ``response.output`` from its recorded token ids.

    Call this after the harness and verifier finish the record.
    The function mutates ``result`` in place.
    It replaces only ``response.output``.
    It preserves the reward and all other harness and verifier output.

    The function freezes capture records through ``source``.
    It rebuilds from that frozen snapshot.
    It never retires the snapshot.
    A ``None`` source means this caller does not rebuild.
    A rollout that already carries token ids is left unchanged.
    Its redundant frozen capture remains eligible for retirement after handoff.

    The function never raises.
    Missing or ambiguous tokens cause masking.
    Failed or masked builds retain their frozen evidence.

    Return the build with its rebuilt response, metrics, and optional error.
    Return ``None`` when no source exists.
    An unusable build has no rebuilt response and sets ``mask_sample``.
    """
    if source is None:
        return None

    try:
        rollout_id = maybe_rollout_id_from_run_body(result)
    except (TypeError, ValueError) as error:
        # A malformed explicit id cannot be looked up.
        # Mask the rollout instead of raising.
        return _unusable(
            result,
            f"malformed rollout id: {error}",
            f"a rollout result carries a malformed id ({error}), so its recorded token ids could "
            "not be looked up and it will be token-less.",
        )
    if rollout_carries_token_ids(result):
        # Re-finalization must return the frozen snapshot.
        # The caller must be able to retire on every path.
        if rollout_id is None:
            return None
        try:
            snapshot = await source.freeze(rollout_id)
        except Exception:
            warnings.warn(f"could not freeze redundant records for rollout {rollout_id}.", stacklevel=2)
            return None
        return {
            "rebuilt_response": None,
            MASK_SAMPLE_KEY: False,
            _REDUNDANT_CAPTURE_KEY: True,
            "_capture_snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "version": snapshot.version,
            },
        }

    if rollout_id is None:
        # No correlation key was preserved on the finished record.
        return _unusable(
            result,
            "no capture key",
            "a rollout result carries no id and no task/rollout indices, so its recorded token ids "
            "could not be looked up and it will be token-less.",
        )

    response = result.get("response") if isinstance(result.get("response"), dict) else {}
    explicit_terminal = result.get(TERMINAL_CALL_KEY)
    try:
        # The result's response is what the verifier scored.
        # Terminal attribution joins it to one captured call.
        built = await trajectories_from_source(
            rollout_id,
            source,
            model=str(response.get("model") or ""),
            verified_response=response or None,
            explicit_terminal_call_id=str(explicit_terminal) if explicit_terminal else None,
        )
    except Exception as error:
        # A transport failure may be unrelated to this rollout.
        # Mask this rollout instead of failing the entire batch.
        return _unusable(
            result,
            f"{type(error).__name__}: {error}",
            f"could not read the records for rollout {rollout_id}: {type(error).__name__}: {error}. "
            "It will be token-less.",
        )
    if built is None:
        # Correlation failed between the agent and capture middleware.
        # An external harness or proxy may have dropped ``/ng-rollout/<id>/training-token-capture``.
        return _unusable(
            result,
            "nothing recorded",
            f"rollout {rollout_id} has no token ids and none were recorded for it, so it was not "
            "rebuilt and will be token-less. Its model calls likely did not reach the capture "
            "middleware correlated.",
        )

    projected = built["rebuilt_response"]
    if projected is not None:
        if isinstance(result.get("response"), dict):
            result["response"]["output"] = projected["output"]
        else:
            result["response"] = projected

    # Record build losses so partial trajectories remain visible.
    record_metrics = dict(built.get("metrics") or {})
    if built.get("error"):
        record_metrics["error"] = built["error"]
    if built.get(MASK_SAMPLE_KEY):
        # Keep the masking verdict at the top level.
        # Retain its reasons in the metrics.
        result[MASK_SAMPLE_KEY] = True
        warnings.warn(
            f"rollout {rollout_id} was captured incompletely or ambiguously ({record_metrics}); "
            "it is marked for masking rather than trained on.",
            stacklevel=2,
        )
    result[TOKEN_CAPTURE_KEY] = record_metrics

    return built


async def retire_rollout_token_capture(
    rollout_id: str,
    source: TokenSource | None,
    built: dict | None,
) -> bool:
    """Retire a frozen snapshot after durable handoff.

    The caller owns the durability boundary.
    Call this only after downstream acceptance or a local fsync.
    Retirement uses the frozen ``snapshot_id`` and version.
    Failed or masked builds remain as diagnostic evidence.
    """
    if source is None or not capture_build_can_retire(built):
        return False
    snapshot = built.get("_capture_snapshot")
    if not isinstance(snapshot, dict):
        warnings.warn(f"rollout {rollout_id} has no frozen capture identity to retire.", stacklevel=2)
        return False
    try:
        return await source.drop(
            rollout_id,
            snapshot_id=str(snapshot["snapshot_id"]),
            version=int(snapshot["version"]),
        )
    except Exception:
        warnings.warn(f"could not retire the records for rollout {rollout_id}.", stacklevel=2)
        return False


def capture_build_can_retire(built: dict | None) -> bool:
    """Whether a successful build consumed a frozen snapshot."""
    if built is None or built.get(MASK_SAMPLE_KEY):
        return False
    return built.get("rebuilt_response") is not None or bool(built.get(_REDUNDANT_CAPTURE_KEY))
