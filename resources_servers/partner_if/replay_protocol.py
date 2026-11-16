# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PartnerIF's private replay-review wire contract; the Gym episode protocol stays unchanged."""

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_serializer, model_validator

from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.trajectory import VerifierInvocation


REPLAY_RECEIPT_KEY = "partner_if_replay_receipt"


class ReplayReceiptReference(BaseModel):
    """An opaque Resources-owned receipt, bound to the submitted review request."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PrefixReviewRequest(BaseModel):
    """One immutable prefix and the complete original task supplied to its agent session."""

    model_config = ConfigDict(extra="forbid")
    episode_id: EpisodeId
    task_id: TaskId
    agent_session_id: str = Field(min_length=1)
    prefix_index: int = Field(ge=1, le=32)
    task_data: dict[str, JsonValue]
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse

    @field_serializer("responses_create_params")
    def serialize_actor_params(self, value: NeMoGymResponseCreateParamsNonStreaming):
        """Do not introduce typed defaults into the frozen actor request."""
        return value.model_dump(mode="json", exclude_unset=True, exclude_none=True)


class PrefixReviewResponse(BaseModel):
    """Resources owns the grade, terminal masking, and the next bounded prefix decision."""

    # Preserve the shared verifier's full judge and failure receipts without creating a second grade schema.
    model_config = ConfigDict(extra="allow")
    response: NeMoGymResponse
    reward: float
    mask_sample: bool
    evaluation_completed: bool
    next_parent_turn_limit: int | None = Field(default=None, ge=2, le=32)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    ng_verifier_invocations: list[VerifierInvocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def receipt_matches_response(self) -> Self:
        reference = read_replay_receipt(self.response)
        if reference.id != self.receipt_id or reference.request_sha256 != self.request_sha256:
            raise ValueError("Canonical response and review receipt identity disagree")
        return self


def json_sha256(value: object) -> str:
    """Hash JSON values independent of transport whitespace and mapping insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def replay_request_sha256(request: PrefixReviewRequest) -> str:
    """Bind all submitted task, runtime, capture, and cumulative response fields."""
    return json_sha256(request.model_dump(mode="json"))


def read_replay_receipt(response: NeMoGymResponse) -> ReplayReceiptReference:
    """Validate a receipt reference without trusting arbitrary response extras."""
    return ReplayReceiptReference.model_validate((response.model_extra or {}).get(REPLAY_RECEIPT_KEY))


def canonical_replay_response(
    response: NeMoGymResponse, receipt_id: str, request_sha256: str, *, masked: bool = False
) -> NeMoGymResponse:
    """Add the receipt and, for inconclusive reviews, mark status incomplete before final binding.

    Output, model identity, usage, and immutable replay diagnostics are never changed by Resources.
    """
    reference = ReplayReceiptReference(id=receipt_id, request_sha256=request_sha256)
    updates: dict[str, object] = {REPLAY_RECEIPT_KEY: reference.model_dump(mode="json")}
    if masked:
        updates["status"] = "incomplete"
    return NeMoGymResponse.model_validate(response.model_dump(mode="json") | updates)
