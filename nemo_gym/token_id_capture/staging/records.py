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

"""Staging wire shapes (gate-authoritative capture; single definition).

These are the records the token-in/token-out pipeline moves between the
inference worker, the framework's ``TokenSink`` (staging storage), the gate,
and the RL controller. They extend #2124's ``TokenEntry`` world: ``TokenEntry``
(``token_id_capture/records.py``) remains the read-route record rebuilt for
trainers; the shapes below are what travels while a rollout is in flight.
Shapes freeze at the S1 review gate; the optional ``chain_hash``/``cum_hash``
fields are reserved so the hardening layer (H2) is purely additive.

This module is part of the dependency-free capture core (the ``staging``
subpackage purity rule): stdlib + pydantic only.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = 1

CaptureDisposition = Literal["staged", "capture_failed"]
CaptureMode = Literal["token_in", "text"]


def staging_key(rollout_id: str, call_id: str) -> str:
    """The storage key for one staged call delta.

    Opaque to Gym beyond construction: frameworks may key TQ rows, file
    paths, or redis entries with it. The receipt manifest is the only join
    between lineage and storage.
    """
    return f"{rollout_id}/{call_id}"


class StagedCallRecord(BaseModel):
    """One model call's token delta, staged by the worker to the framework's
    ``TokenSink`` -- the design's only heavy hop.

    ``token_ids_delta`` is ``rendered_prompt[prev_len:] + generated`` with
    ``token_mask_delta`` 0.0 on the carried prompt and 1.0 on generated
    tokens; ``generation_logprobs_delta`` is 0.0 on the carry. ``digest`` is
    ``compute_staging_digest`` over the delta arrays and identity fields.
    """

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    call_id: str
    parent_call_id: Optional[str] = None
    prev_len: int
    new_len: int
    weight_version: int
    digest: str
    schema_version: int = SCHEMA_VERSION
    token_ids_delta: list[int]
    token_mask_delta: list[float]
    generation_logprobs_delta: list[float]
    # Optional per-token extras (e.g. MoE ``routed_experts``), staged beside
    # the delta so they never transit gate HTTP.
    extras: Optional[dict[str, Any]] = None
    # Reserved for H2 hardening; absent in MVP records.
    chain_hash: Optional[str] = None
    cum_hash: Optional[str] = None

    @property
    def staging_key(self) -> str:
        return staging_key(self.rollout_id, self.call_id)


class StageResult(BaseModel):
    """What a ``TokenSink.stage`` call reports back to the capture layer."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    staging_key: str
    error: Optional[str] = None


class CommitCoords(BaseModel):
    """The token-light receipt a worker returns to the gate on the response.

    Ingesting these coordinates IS the authoritative commit: the gate extends
    its per-rollout token buffer with ``token_ids_delta`` and records the call
    in lineage. ``disposition == "capture_failed"`` means the completion is
    still served but the rollout is capture-poisoned.
    """

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    call_id: str
    parent_call_id: Optional[str] = None
    delta_len: int
    cum_len: int
    digest: str
    staging_key: str
    weight_version: int
    disposition: CaptureDisposition = "staged"
    schema_version: int = SCHEMA_VERSION
    # Delta ids for the gate's buffer (~4 B/token; never logprobs).
    token_ids_delta: list[int] = Field(default_factory=list)


class CallRecord(BaseModel):
    """One committed call in a sealed rollout's manifest (token-free)."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    parent_call_id: Optional[str] = None
    delta_len: int
    cum_len: int
    digest: str
    staging_key: str
    weight_version: int
    mode: CaptureMode = "token_in"
    # Reserved for H2 hardening; absent in MVP records.
    chain_hash: Optional[str] = None
    cum_hash: Optional[str] = None


class RolloutReceipt(BaseModel):
    """The token-free record the gate returns to the controller at seal.

    The manifest lists committed calls in admission order; ``terminal_call_id``
    names the call whose chain is the training row (the linearizer's
    ``terminal_hint``). ``capture_poisoned`` is set when any call staged with
    ``disposition == "capture_failed"`` -- the finalizer produces a placeholder.
    """

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    reward: Optional[float] = None
    terminal_call_id: Optional[str] = None
    manifest: list[CallRecord] = Field(default_factory=list)
    capture_poisoned: bool = False
    failure_reason: Optional[str] = None
    schema_version: int = SCHEMA_VERSION


class StagedCallSnapshot(BaseModel):
    """One verified staging row, as the finalizer's ``TokenSource`` returns it.

    ``prev_len`` is the length of the committed sequence this delta extends
    (0 for roots); the delta layout matches ``StagedCallRecord``. Rows carry
    an explicit storage parent; a rootless row has none and is self-contained
    (``prev_len == 0``).
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str
    prev_len: int
    token_ids_delta: list[int]
    token_mask_delta: list[float]
    logprobs_delta: list[float]
    weight_version: Optional[int] = None
    parent_call_id: Optional[str] = None
    model: str = ""
