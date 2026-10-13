# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire contract for cutting in-flight generations at a checkpoint.

A generation cut keeps the tokens an in-flight policy call has already generated.
The model server sends each inference worker the inventory of calls it is serving;
the worker freezes each call's generated prefix,
stages it durably in the training framework's token store, and returns a receipt.
Returning the receipt is the worker's durability boundary.

On restore, the replacement attempt re-issues the call from the agent's last boundary.
Its capture admission carries a ``GenerationCutContinuation`` naming the staged prefix,
and the worker continues generating after that prefix instead of starting over.

Cuts only preserve work.
A call without a durable prefix is regenerated after restore, which is always correct,
so a worker without cut support or a failed cut never blocks a checkpoint.

Field names match the worker endpoint that NeMo-RL implements at ``/ng-control/v1/generation-cut``.
"""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nemo_gym.episode_types import EpisodeId


GENERATION_CUT_ROUTE = "/ng-control/v1/generation-cut"
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class _GenerationCutModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationCutPrefix(_GenerationCutModel):
    """One admitted call whose generated prefix needs a durable cut."""

    ticket_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1, pattern=_ID_PATTERN)
    attempt: int = Field(ge=0)
    model_call_id: str = Field(min_length=1)
    admitted_at: float


class GenerationCutInventory(_GenerationCutModel):
    """The calls one worker must cut, bound to one checkpoint by digest."""

    checkpoint_id: str = Field(min_length=1)
    server_name: str = Field(min_length=1)
    active_prefixes: tuple[GenerationCutPrefix, ...] = ()
    inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def build(
        cls, *, checkpoint_id: str, server_name: str, active_prefixes: list[GenerationCutPrefix]
    ) -> "GenerationCutInventory":
        prefixes = tuple(sorted(active_prefixes, key=lambda prefix: prefix.ticket_id))
        payload = {
            "checkpoint_id": checkpoint_id,
            "server_name": server_name,
            "active_prefixes": [prefix.model_dump(mode="json") for prefix in prefixes],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(**payload, inventory_digest=digest)


class GenerationCutPrefixAck(_GenerationCutModel):
    """The worker's durable outcome for one call in the inventory."""

    ticket_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1, pattern=_ID_PATTERN)
    attempt: int = Field(ge=0)
    model_call_id: str = Field(min_length=1)
    admitted_at: float
    disposition: Literal["durable_prefix", "durable_failure"]
    cut_kind: Literal["active_prefix", "terminal_completion"] | None = None
    frozen_buffer_id: str | None = Field(default=None, min_length=1)
    staging_keys: tuple[str, ...] = ()
    prefix_token_count: int | None = Field(default=None, ge=0)
    prefix_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    effective_output_limit: int | None = Field(default=None, ge=1)
    terminal_finish_reason: Literal["stop", "length"] | None = None
    terminal_stop_reason: str | int | None = None

    @model_validator(mode="after")
    def _validate_prefix_evidence(self) -> Self:
        evidence = (self.cut_kind, self.frozen_buffer_id, self.prefix_token_count, self.prefix_digest)
        if self.disposition == "durable_prefix":
            if (
                any(value is None for value in evidence)
                or not self.staging_keys
                or self.effective_output_limit is None
            ):
                raise ValueError(
                    "durable_prefix requires cut_kind, frozen_buffer_id, staging_keys, prefix_token_count, "
                    "prefix_digest, and effective_output_limit"
                )
        elif any(value is not None for value in evidence) or self.staging_keys or self.effective_output_limit:
            raise ValueError("durable_failure cannot carry prefix evidence")
        if self.terminal_stop_reason is not None and self.terminal_finish_reason is None:
            raise ValueError("terminal_stop_reason requires terminal_finish_reason")
        if len(self.staging_keys) != len(set(self.staging_keys)):
            raise ValueError("generation-cut staging_keys must be unique")
        return self

    @classmethod
    def failure(cls, prefix: GenerationCutPrefix) -> "GenerationCutPrefixAck":
        return cls(**prefix.model_dump(), disposition="durable_failure")

    @property
    def episode_id(self) -> EpisodeId:
        return EpisodeId(rollout_id=self.rollout_id, attempt=self.attempt)


class GenerationCutReceipt(_GenerationCutModel):
    """A worker's final durable receipt for every call in its inventory."""

    checkpoint_id: str = Field(min_length=1)
    cut_id: str = Field(min_length=1)
    inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory: GenerationCutInventory
    backend_snapshot_id: str = Field(min_length=1)
    prefixes: tuple[GenerationCutPrefixAck, ...] = ()

    def validate_for(self, inventory: GenerationCutInventory) -> None:
        """Reject a receipt that is not an exact, complete answer to ``inventory``."""
        if self.inventory != inventory or self.inventory_digest != inventory.inventory_digest:
            raise ValueError("generation-cut receipt answers a different inventory")
        admitted = {prefix.ticket_id: prefix for prefix in inventory.active_prefixes}
        acked = [prefix.ticket_id for prefix in self.prefixes]
        if len(acked) != len(set(acked)) or set(acked) != set(admitted):
            raise ValueError("generation-cut receipt must cover every call in the inventory exactly once")
        for ack in self.prefixes:
            call = admitted[ack.ticket_id]
            if (ack.rollout_id, ack.attempt, ack.model_call_id, ack.admitted_at) != (
                call.rollout_id,
                call.attempt,
                call.model_call_id,
                call.admitted_at,
            ):
                raise ValueError(f"generation-cut receipt identity differs for ticket {ack.ticket_id!r}")
