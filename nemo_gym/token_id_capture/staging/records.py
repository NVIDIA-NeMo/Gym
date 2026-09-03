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

"""Versioned wire records for framework-owned token staging."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator

from nemo_gym.token_id_capture.records import (  # noqa: F401  (re-exported for staging consumers)
    LEDGER_ROW_MISSING_RESPONSE_ID_REASON,
    UNCOMMITTED_CALL_REASON,
    UNRESOLVED_PARENT_REASON,
)
from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_EXTRAS_DIGEST,
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    compute_extras_digest,
    compute_staging_digest,
)


CaptureDisposition = Literal["staged", "capture_failed"]
CaptureMode = Literal["token_in", "text"]
Identifier = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
DigestHex = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


def staging_key(rollout_id: str, model_call_id: str) -> str:
    """Return the default deterministic key used by the NeMo RL backend."""
    if not rollout_id or not model_call_id:
        raise ValueError("rollout_id and model_call_id must be non-empty")
    return f"{rollout_id}/{model_call_id}"


# --- reason-code vocabulary ------------------------------------------------
# The complete set of machine-readable reasons a manifest can carry. Poison
# reasons ride ``ManifestFailure.reason`` / ``RolloutReceipt.failure_reason``
# and fail linearization closed; selection reasons ride
# ``TerminalSelection.reason`` from the parent-link fallback walk. Values are
# wire contract: the framework switches on them, so never repurpose one.

# Reasons specific to worker commit coordinates. The shared reasons
# (``UNRESOLVED_PARENT_REASON``, ``UNCOMMITTED_CALL_REASON``,
# ``LEDGER_ROW_MISSING_RESPONSE_ID_REASON``) live in the core records module
# and are re-exported here.

# The worker's response carried no ``ng_commit_coords`` acknowledgement.
WORKER_MISSING_COMMIT_COORDS_REASON = "worker_response_missing_commit_coordinates"
# The worker acknowledged the call with ``disposition="capture_failed"``.
WORKER_CAPTURE_FAILED_REASON = "worker_capture_failed"
# The worker's commit coordinates failed identity validation against the
# admitted capture context.
INVALID_COMMIT_COORDS_REASON = "invalid_worker_commit_coordinates"

# Terminal-selection outcomes from the no-witness parent-link walk
# (``staging.terminal.select_terminal_call``).
TERMINAL_SELECTED = "selected"
TERMINAL_NO_RECORDS = "no_records"
TERMINAL_DUPLICATE_CALL_ID = "duplicate_call_id"
TERMINAL_ORPHANED_ROW = "orphaned_row"
TERMINAL_NO_ROOT = "no_root"
TERMINAL_AMBIGUOUS = "ambiguous_terminal"


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[STAGING_SCHEMA_VERSION] = STAGING_SCHEMA_VERSION


class _DigestWireModel(_WireModel):
    digest_version: Literal[STAGING_DIGEST_VERSION] = STAGING_DIGEST_VERSION
    extras_digest_version: Literal[EXTRAS_DIGEST_VERSION] = EXTRAS_DIGEST_VERSION


class CaptureAdmission(_WireModel):
    """Gate-to-worker identity and exact-prefix contract for one model call.

    A ``token_in`` admission carries its prefix in one of two encodings.
    ``required_prefix_token_ids`` holds the ids inline.
    ``staging_chain`` names the staged records whose token deltas form the prefix.
    The worker resolves either encoding to a flat id list.
    It then passes that list to ``CaptureAdapter.enter_prefix``.
    """

    rollout_id: Identifier
    model_call_id: Identifier
    parent_call_id: Identifier | None = None
    prev_len: NonNegativeInt = 0
    mode: CaptureMode
    required_prefix_token_ids: list[StrictInt] = Field(default_factory=list)
    staging_chain: list[str] = Field(default_factory=list)
    parent_chain_hash: DigestHex | None = None

    @model_validator(mode="after")
    def _validate_prefix_contract(self) -> Self:
        if any(token_id < 0 for token_id in self.required_prefix_token_ids):
            raise ValueError("required_prefix_token_ids must be non-negative")
        if self.mode == "token_in":
            if self.parent_call_id is None or self.prev_len == 0:
                raise ValueError("token_in admission requires a parent_call_id and prev_len > 0")
            if self.parent_chain_hash is None:
                raise ValueError("token_in admission requires the parent's chain hash")
            # When staging_chain is supplied the worker fetches the prefix from TQ;
            # required_prefix_token_ids is allowed to be empty on the wire.
            if not self.staging_chain and len(self.required_prefix_token_ids) != self.prev_len:
                raise ValueError("required_prefix_token_ids length must equal prev_len")
        elif (
            self.parent_call_id is not None
            or self.prev_len != 0
            or self.required_prefix_token_ids
            or self.staging_chain
            or self.parent_chain_hash is not None
        ):
            raise ValueError("text admission must be a parentless root with no required prefix")
        return self


class StagedCallBaseSnapshot(_DigestWireModel):
    """Token columns and metadata for one staged call.

    This model excludes the optional ``extras`` payload.
    Validation recomputes the call digest from these fields and the recorded ``extras_digest``.
    It does not verify that the corresponding extras payload exists or matches that digest.
    A consumer that loads extras must verify them separately with ``compute_extras_digest``.
    """

    rollout_id: Identifier
    model_call_id: Identifier
    parent_call_id: Identifier | None = None
    mode: CaptureMode
    prev_len: NonNegativeInt
    delta_len: NonNegativeInt
    cum_len: NonNegativeInt
    weight_version: NonNegativeInt
    digest: DigestHex
    token_ids_delta: list[StrictInt]
    token_mask_delta: list[StrictFloat]
    generation_log_probs_delta: list[StrictFloat]
    extras_digest: DigestHex = EMPTY_EXTRAS_DIGEST
    chain_hash: DigestHex | None = None
    cumulative_hash: DigestHex | None = None

    @property
    def staging_key(self) -> str:
        return staging_key(self.rollout_id, self.model_call_id)

    @model_validator(mode="after")
    def _validate_base_integrity(self) -> Self:
        if self.parent_call_id is None and self.prev_len != 0:
            raise ValueError("a parentless staged call must have prev_len == 0")
        if self.parent_call_id is not None and self.prev_len == 0:
            raise ValueError("a child staged call must have prev_len > 0")
        if self.parent_call_id is None and self.mode != "text":
            raise ValueError("a parentless staged call must use text mode")
        if self.parent_call_id is not None and self.mode != "token_in":
            raise ValueError("a child staged call must use token_in mode")
        if self.delta_len == 0:
            raise ValueError("a staged call delta must contain at least one token")
        if any(token_id < 0 for token_id in self.token_ids_delta):
            raise ValueError("token_ids_delta must be non-negative")
        if any(mask not in (0.0, 1.0) for mask in self.token_mask_delta):
            raise ValueError("token_mask_delta must contain only 0.0 or 1.0")
        if any(
            mask == 0.0 and log_prob != 0.0
            for mask, log_prob in zip(self.token_mask_delta, self.generation_log_probs_delta)
        ):
            raise ValueError("prompt-carry log probabilities must be 0.0")
        actual_digest = compute_staging_digest(
            schema_version=self.schema_version,
            digest_version=self.digest_version,
            extras_digest_version=self.extras_digest_version,
            rollout_id=self.rollout_id,
            model_call_id=self.model_call_id,
            parent_call_id=self.parent_call_id,
            mode=self.mode,
            prev_len=self.prev_len,
            delta_len=self.delta_len,
            cum_len=self.cum_len,
            weight_version=self.weight_version,
            token_ids_delta=self.token_ids_delta,
            token_mask_delta=self.token_mask_delta,
            generation_log_probs_delta=self.generation_log_probs_delta,
            extras_digest=self.extras_digest,
            chain_hash=self.chain_hash,
            cumulative_hash=self.cumulative_hash,
        )
        if self.digest != actual_digest:
            raise ValueError("digest does not match staged call contents")
        return self


class StagedCallRecord(StagedCallBaseSnapshot):
    """One normalized token delta made durable by a framework staging sink."""

    extras: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_extras_integrity(self) -> Self:
        actual_extras_digest = compute_extras_digest(self.extras)
        if self.extras_digest != actual_extras_digest:
            raise ValueError("extras_digest does not match extras")
        return self


class StagedCallSnapshot(StagedCallRecord):
    """The exact staged record returned by a framework-owned source."""


class StageResult(_WireModel):
    """Durability result returned by ``StagingSink.stage``."""

    ok: bool
    staging_key: Identifier | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.ok and self.staging_key is None:
            raise ValueError("a successful stage result requires staging_key")
        if not self.ok and not self.error:
            raise ValueError("a failed stage result requires an error")
        return self


class CommitCoords(_DigestWireModel):
    """Metadata returned after an inference worker stages one call."""

    rollout_id: Identifier
    model_call_id: Identifier
    parent_call_id: Identifier | None = None
    prev_len: NonNegativeInt
    delta_len: NonNegativeInt
    cum_len: NonNegativeInt
    weight_version: NonNegativeInt
    disposition: CaptureDisposition = "staged"
    digest: DigestHex | None = None
    extras_digest: DigestHex | None = None
    staging_key: Identifier | None = None
    chain_hash: DigestHex | None = None
    cumulative_hash: DigestHex | None = None

    @model_validator(mode="after")
    def _validate_disposition(self) -> Self:
        if self.parent_call_id is None and self.prev_len != 0:
            raise ValueError("parentless coordinates must have prev_len == 0")
        if self.parent_call_id is not None and self.prev_len == 0:
            raise ValueError("child coordinates must have prev_len > 0")
        if self.cum_len != self.prev_len + self.delta_len:
            raise ValueError("cum_len must equal prev_len + delta_len")
        staged_payload = (self.digest, self.extras_digest, self.staging_key, self.chain_hash, self.cumulative_hash)
        if self.disposition == "staged":
            if any(value is None for value in staged_payload):
                raise ValueError(
                    "staged coordinates require digest, extras_digest, staging_key, chain_hash, and cumulative_hash"
                )
            if self.delta_len == 0:
                raise ValueError("staged coordinates require a non-empty delta")
        elif any(value is not None for value in staged_payload):
            raise ValueError("capture_failed coordinates cannot carry staged payload metadata")
        elif self.delta_len != 0:
            raise ValueError("capture_failed coordinates cannot carry token deltas")
        return self


class CallRecord(_DigestWireModel):
    """One token-free call manifest row in a rollout's capture ledger."""

    model_call_id: Identifier
    parent_call_id: Identifier | None = None
    prev_len: NonNegativeInt
    delta_len: NonNegativeInt
    cum_len: NonNegativeInt
    weight_version: NonNegativeInt
    digest: DigestHex
    extras_digest: DigestHex
    staging_key: Identifier
    mode: CaptureMode = "token_in"
    chain_hash: DigestHex | None = None
    cumulative_hash: DigestHex | None = None
    # The response envelope ID observed before the model server returns the response.
    # Terminal attribution matches the scored ``response.id`` to this field.
    response_id: Identifier
    # Wall-clock admission time stamped by the model-server middleware.
    # Heuristic terminal selection orders candidate roots by this value.
    admitted_at: StrictFloat | None = None
    # Fingerprints of this call output and of the request followed by that output.
    # A value is ``None`` when the corresponding items cannot be fingerprinted.
    output_fingerprint: DigestHex | None = None
    continuation_fingerprint: DigestHex | None = None
    # Canonicalization version of the fingerprints above; 0 means none were
    # recorded. Attribution ignores fingerprints from a different version.
    fingerprint_version: NonNegativeInt = 0

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        if self.delta_len == 0 or self.cum_len != self.prev_len + self.delta_len:
            raise ValueError("manifest lengths must describe a non-empty contiguous delta")
        if self.parent_call_id is None and self.prev_len != 0:
            raise ValueError("a parentless manifest row must have prev_len == 0")
        if self.parent_call_id is not None and self.prev_len == 0:
            raise ValueError("a child manifest row must have prev_len > 0")
        if self.parent_call_id is None and self.mode != "text":
            raise ValueError("a parentless manifest row must use text mode")
        if self.parent_call_id is not None and self.mode != "token_in":
            raise ValueError("a child manifest row must use token_in mode")
        return self


class CaptureLedgerCommit(_WireModel):
    """One successfully staged call, as handed to ``CaptureLedger.record``.

    ``record`` is the manifest row. The remaining fields feed parent resolution
    and are not part of the manifest.
    """

    rollout_id: Identifier
    record: CallRecord
    staging_chain: tuple[Identifier, ...] = ()
    request_items: list[dict]
    response_items: list[dict]


class ManifestFailure(_WireModel):
    """A call whose token capture did not complete."""

    model_call_id: Identifier
    reason: Identifier


class RolloutManifest(_WireModel):
    """The token-free ledger read returned by the manifest control route."""

    rollout_id: Identifier
    records: list[CallRecord] = Field(default_factory=list)
    failures: list[ManifestFailure] = Field(default_factory=list)


class RolloutReceipt(_DigestWireModel):
    """Token-free immutable manifest the framework assembles at rollout end."""

    rollout_id: Identifier
    reward: float | None = None
    terminal_model_call_id: Identifier | None = None
    manifest: list[CallRecord] = Field(default_factory=list)
    capture_poisoned: bool = False
    failure_reason: str | None = None
    # How ``terminal_model_call_id`` was chosen: ``declared`` when the agent
    # named the response it kept, ``response_id``/``content`` when a witness
    # joined the scored response to a manifest row, ``heuristic`` when the
    # parent-link walk inferred it (also stamped on failed selections — it
    # names the last stage attempted).
    terminal_selection: Literal["declared", "response_id", "content", "heuristic"]
    # The witness abstention/corroboration trail from attribution, kept on
    # success and failure alike so per-method metrics stay diagnosable.
    terminal_attribution_reason: str | None = None

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        model_call_ids = [record.model_call_id for record in self.manifest]
        if len(model_call_ids) != len(set(model_call_ids)):
            raise ValueError("receipt manifest contains duplicate model_call_id values")
        if self.terminal_model_call_id is not None and self.terminal_model_call_id not in set(model_call_ids):
            raise ValueError("terminal_model_call_id is absent from the receipt manifest")
        return self
