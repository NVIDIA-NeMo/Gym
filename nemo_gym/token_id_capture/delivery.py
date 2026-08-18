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

"""Turning one finished rollout record into a token-bearing one.

This is the step between capture and training: the rollout has run, its model
calls were recorded somewhere, and the record needs its ``response.output``
replaced with the merged trajectory before anything reads it.

It takes the record and a ``TokenSource``, and nothing else. Both belong to the
caller. Every configuration question, whether capture is on, where records were
written, which agents emit the correlation prefix, is answered on whichever side
owns that configuration and never read across the boundary: Gym's rollout
collection resolves its source from Gym's config, and a training framework
staging records through its own data plane passes its own source and never
touches Gym's config at all.

What a rollout needs is read from the rollout. An item that already carries
token ids holds what the policy sampled, so that rollout is left alone; an item
without them needs the recorded ids attached. That works for a batch mixing
native agents and external harnesses, and keeps working as native agents move
onto capture: such an agent simply stops arriving with ids and starts being
rebuilt, with no change here.
"""

from __future__ import annotations

import warnings

from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.token_id_capture.consumer import trajectories_from_source
from nemo_gym.token_id_capture.protocols import TokenSource


# Per-rollout token-capture health, attached to the record so a run can see what the build dropped:
# chain count, quarantined fraction, delivered token fraction, and why a rollout was masked.
TOKEN_CAPTURE_KEY = "_ng_token_capture"

# The one field a consumer reads to decide whether to drop this rollout from the loss. It sits at
# the top of the record, not inside TOKEN_CAPTURE_KEY, so nothing has to know this feature exists
# to find it. The name is the one already used for the same request elsewhere in the repo.
MASK_SAMPLE_KEY = "mask_sample"


def rollout_carries_token_ids(result: dict) -> bool:
    """Whether this rollout already holds what training needs.

    True when any output item carries generated token ids. Those are what the policy actually
    sampled, so they win over anything rebuilt from records: a reconstruction can differ, and
    overwriting them would train on the difference without saying so. "Any" rather than "all" for
    that reason, since a partly covered rollout is a problem to look at rather than one to paper
    over with a rebuild.
    """
    response = result.get("response")
    if not isinstance(response, dict):
        return False
    return any(isinstance(item, dict) and item.get("generation_token_ids") for item in (response.get("output") or []))


def _unusable(result: dict, error: str, message: str) -> dict:
    """Mask a rollout that needed token ids and could not get them.

    Unmasked it reaches the trainer looking healthy and fails there instead, further from the
    cause. The reason goes on the record so a run can count these rather than read logs for them.
    """
    warnings.warn(message, stacklevel=3)
    metrics = {"n_calls": 0, "error": error}
    result[MASK_SAMPLE_KEY] = True
    result[TOKEN_CAPTURE_KEY] = metrics
    return {"rebuilt_response": None, MASK_SAMPLE_KEY: True, "error": error, "metrics": metrics}


async def finalize_rollout_token_capture(result: dict, source: TokenSource | None) -> dict | None:
    """Rebuild one finished rollout record's ``response.output`` from its recorded token ids.

    Call this once per record, after the harness and verifier are done with it. Mutates ``result``
    in place. The reward, the reward components and everything else the harness and verifier
    produced are left alone; only ``response.output`` is replaced, so a trainer reads an external
    harness's rollout exactly as it reads a native agent's, with no sidecar payload and no
    per-agent branch.

    ``source`` is where records are read from and retired, and ``None`` means this caller is not
    capturing, so there is nothing to do. Idempotent: a rollout already rebuilt carries token ids
    and is left alone on a second call.

    Never raises. A rollout whose tokens are missing or ambiguous is marked for masking, since the
    alternative is training on a trajectory with a hole in it.

    Returns the build (its ``rebuilt_response``, ``metrics`` and any ``error``), or ``None`` when
    there was nothing to do: no source, or a rollout that already carries ids. A rollout that
    needed ids and could not get them returns a build with no ``rebuilt_response`` and
    ``mask_sample`` set, so a caller counting across a batch sees it as masked and unbuilt, which
    cannot be recovered from the record afterwards.
    """
    if source is None or rollout_carries_token_ids(result):
        return None

    rollout_id = maybe_rollout_id_from_run_body(result)
    if rollout_id is None:
        # Nothing to look records up by. Usually the correlation key was not carried onto the
        # finished record.
        return _unusable(
            result,
            "no capture key",
            "a rollout result carries no id and no task/rollout indices, so its recorded token ids "
            "could not be looked up and it will be token-less.",
        )

    response = result.get("response") if isinstance(result.get("response"), dict) else {}
    try:
        built = await trajectories_from_source(rollout_id, source, model=str(response.get("model") or ""))
    except Exception as error:
        # A transport can fail for reasons unrelated to this rollout. Raising here would take down
        # the caller's whole batch, so lose the one rollout instead.
        return _unusable(
            result,
            f"{type(error).__name__}: {error}",
            f"could not read the records for rollout {rollout_id}: {type(error).__name__}: {error}. "
            "It will be token-less.",
        )
    if built is None:
        # Correlation broke between the agent and the capture middleware, such as an external
        # harness or proxy that did not preserve the /ng-rollout prefix.
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

    # Carry build losses onto the rollout record.
    # Otherwise a partial trajectory looks like one that retained every call.
    record_metrics = dict(built.get("metrics") or {})
    if built.get("error"):
        record_metrics["error"] = built["error"]
    if built.get(MASK_SAMPLE_KEY):
        # Keep the masking verdict at the top of the record.
        # The metrics dictionary retains its reasons.
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
    """Conditionally retire a snapshot after its rebuilt rollout is durably handed off.

    The caller owns the durability boundary.
    Call this only after downstream acceptance or a local fsync.
    Failed or masked builds remain as diagnostic evidence.
    """
    if source is None or built is None or built.get("rebuilt_response") is None or built.get(MASK_SAMPLE_KEY):
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
