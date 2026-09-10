# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Typed contracts shared by model admission participants and cut backends."""

import hashlib
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym._checkpoint.control import CONTROL_URL_PREFIX, CheckpointControlRequest
from nemo_gym.rollout_correlation import ROLLOUT_ID_PATTERN


MODEL_ADMISSION_URL_PREFIX = f"{CONTROL_URL_PREFIX}/model-admission"


class _GenerationCutModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationCutPrefix(_GenerationCutModel):
    """One admitted call whose currently generated prefix needs a durable cut."""

    ticket_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1, pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    model_call_id: str = Field(min_length=1)
    admitted_at: float


class GenerationCutInventory(_GenerationCutModel):
    """Immutable active-prefix membership presented to a framework backend."""

    checkpoint_id: str = Field(min_length=1)
    server_name: str = Field(min_length=1)
    active_prefixes: tuple[GenerationCutPrefix, ...] = ()
    inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def build(
        cls,
        *,
        checkpoint_id: str,
        server_name: str,
        active_prefixes: list[GenerationCutPrefix],
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
    """Durable terminal disposition for one active-prefix ticket."""

    ticket_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1, pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    model_call_id: str = Field(min_length=1)
    admitted_at: float
    disposition: Literal["durable_prefix", "durable_failure"]
    frozen_buffer_id: str | None = Field(default=None, min_length=1)
    staging_key: str | None = Field(default=None, min_length=1)
    prefix_token_count: int | None = Field(default=None, ge=0)
    prefix_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_prefix_evidence(self) -> "GenerationCutPrefixAck":
        evidence = (self.frozen_buffer_id, self.staging_key, self.prefix_token_count, self.prefix_digest)
        if self.disposition == "durable_prefix" and any(value is None for value in evidence):
            raise ValueError(
                "durable_prefix requires frozen_buffer_id, staging_key, prefix_token_count, and prefix_digest"
            )
        if self.disposition == "durable_failure" and any(value is not None for value in evidence):
            raise ValueError("durable_failure cannot carry prefix evidence")
        return self


class GenerationCutReceipt(_GenerationCutModel):
    """Final durable backend receipt for every active prefix in one worker.

    Returning this receipt is the backend's checkpoint durability boundary.
    ``backend_snapshot_id`` identifies recovery state that already survives
    worker loss; there is no second per-worker commit phase.
    """

    checkpoint_id: str = Field(min_length=1)
    cut_id: str = Field(min_length=1)
    inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory: GenerationCutInventory
    backend_snapshot_id: str = Field(min_length=1)
    prefixes: tuple[GenerationCutPrefixAck, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_tickets(self) -> "GenerationCutReceipt":
        if self.checkpoint_id != self.inventory.checkpoint_id:
            raise ValueError("generation-cut ack and embedded inventory checkpoint IDs differ")
        if self.inventory_digest != self.inventory.inventory_digest:
            raise ValueError("generation-cut ack and embedded inventory digests differ")
        ticket_ids = [prefix.ticket_id for prefix in self.prefixes]
        if len(ticket_ids) != len(set(ticket_ids)):
            raise ValueError("generation-cut ack contains duplicate ticket IDs")
        return self

    def validate_for(self, inventory: GenerationCutInventory) -> None:
        if self.checkpoint_id != inventory.checkpoint_id:
            raise ValueError("generation-cut ack belongs to a different checkpoint")
        if self.inventory_digest != inventory.inventory_digest:
            raise ValueError("generation-cut ack inventory digest does not match")
        if self.inventory != inventory:
            raise ValueError("generation-cut ack embedded inventory does not match")
        expected = {prefix.ticket_id for prefix in inventory.active_prefixes}
        actual = {prefix.ticket_id for prefix in self.prefixes}
        if actual != expected:
            raise ValueError("generation-cut ack must cover the complete active-prefix inventory")
        inventory_by_ticket = {prefix.ticket_id: prefix for prefix in inventory.active_prefixes}
        for prefix in self.prefixes:
            admitted = inventory_by_ticket[prefix.ticket_id]
            if (
                prefix.rollout_id != admitted.rollout_id
                or prefix.attempt_index != admitted.attempt_index
                or prefix.model_call_id != admitted.model_call_id
                or prefix.admitted_at != admitted.admitted_at
            ):
                raise ValueError(f"generation-cut ack identity differs for ticket {prefix.ticket_id!r}")


GenerationCutAck = GenerationCutReceipt


class GenerationCutFrozenTicket(_GenerationCutModel):
    """Immutable identity of one ticket frozen by a worker checkpoint cut."""

    ticket_id: str = Field(min_length=1)
    rollout_id: str | None = None
    attempt_index: int | None = Field(default=None, ge=0)
    model_call_id: str | None = None
    generation_started: bool
    response_started: bool

    @model_validator(mode="after")
    def _validate_execution_identity(self) -> "GenerationCutFrozenTicket":
        if (self.rollout_id is None) != (self.attempt_index is None):
            raise ValueError("rollout_id and attempt_index must be provided together")
        return self


class GenerationCutWorkerProof(_GenerationCutModel):
    """One worker's complete frozen membership and durable cut evidence."""

    checkpoint_id: str = Field(min_length=1)
    coordinator_sequence: int = Field(ge=1)
    worker_id: str = Field(min_length=1)
    frozen_tickets: tuple[GenerationCutFrozenTicket, ...] = ()
    ready_ticket_ids: tuple[str, ...] = ()
    generation_pending: int = Field(ge=0)
    membership_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_cut_receipt: GenerationCutReceipt | None = None

    @classmethod
    def build(
        cls,
        *,
        checkpoint_id: str,
        coordinator_sequence: int,
        worker_id: str,
        frozen_tickets: list[GenerationCutFrozenTicket],
        ready_ticket_ids: list[str],
        generation_cut_receipt: GenerationCutReceipt | None,
    ) -> "GenerationCutWorkerProof":
        tickets = tuple(sorted(frozen_tickets, key=lambda ticket: ticket.ticket_id))
        ready = tuple(sorted(ready_ticket_ids))
        payload = {
            "checkpoint_id": checkpoint_id,
            "coordinator_sequence": coordinator_sequence,
            "worker_id": worker_id,
            "frozen_tickets": [ticket.model_dump(mode="json") for ticket in tickets],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(
            **payload,
            ready_ticket_ids=ready,
            generation_pending=len(tickets) - len(ready),
            membership_digest=digest,
            generation_cut_receipt=generation_cut_receipt,
        )

    @model_validator(mode="after")
    def _validate_proof(self) -> "GenerationCutWorkerProof":
        ticket_ids = [ticket.ticket_id for ticket in self.frozen_tickets]
        if ticket_ids != sorted(ticket_ids) or len(ticket_ids) != len(set(ticket_ids)):
            raise ValueError("worker generation-cut membership must contain unique sorted ticket IDs")
        if tuple(sorted(self.ready_ticket_ids)) != self.ready_ticket_ids:
            raise ValueError("worker generation-cut ready ticket IDs must be sorted")
        ready = set(self.ready_ticket_ids)
        frozen = set(ticket_ids)
        if not ready <= frozen:
            raise ValueError("worker generation-cut ready membership contains unknown tickets")
        if self.generation_pending != len(frozen - ready):
            raise ValueError("worker generation-cut pending count does not match frozen membership")
        payload = {
            "checkpoint_id": self.checkpoint_id,
            "coordinator_sequence": self.coordinator_sequence,
            "worker_id": self.worker_id,
            "frozen_tickets": [ticket.model_dump(mode="json") for ticket in self.frozen_tickets],
        }
        expected_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.membership_digest != expected_digest:
            raise ValueError("worker generation-cut membership digest does not match")
        if self.generation_cut_receipt is not None:
            if self.generation_cut_receipt.checkpoint_id != self.checkpoint_id:
                raise ValueError("worker generation-cut receipt belongs to a different checkpoint")
            receipt_tickets = {prefix.ticket_id for prefix in self.generation_cut_receipt.inventory.active_prefixes}
            if not receipt_tickets <= frozen:
                raise ValueError("worker generation-cut receipt contains tickets outside frozen membership")
            if not receipt_tickets <= ready:
                raise ValueError("worker generation-cut receipt tickets are not checkpoint-ready")
            frozen_by_ticket = {ticket.ticket_id: ticket for ticket in self.frozen_tickets}
            for prefix in self.generation_cut_receipt.inventory.active_prefixes:
                ticket = frozen_by_ticket[prefix.ticket_id]
                if (
                    prefix.rollout_id != ticket.rollout_id
                    or prefix.attempt_index != ticket.attempt_index
                    or prefix.model_call_id != ticket.model_call_id
                ):
                    raise ValueError("worker generation-cut receipt identity differs from frozen membership")
        return self


class GenerationCutCoordinatorProof(_GenerationCutModel):
    """Complete worker membership frozen by one coordinator sequence."""

    checkpoint_id: str = Field(min_length=1)
    coordinator_sequence: int = Field(ge=1)
    expected_workers: int = Field(ge=1)
    frozen_worker_ids: tuple[str, ...]
    workers: tuple[GenerationCutWorkerProof, ...]
    proof_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def build(
        cls,
        *,
        checkpoint_id: str,
        coordinator_sequence: int,
        expected_workers: int,
        frozen_worker_ids: tuple[str, ...],
        workers: list[GenerationCutWorkerProof],
    ) -> "GenerationCutCoordinatorProof":
        ordered = tuple(sorted(workers, key=lambda proof: proof.worker_id))
        payload = {
            "checkpoint_id": checkpoint_id,
            "coordinator_sequence": coordinator_sequence,
            "expected_workers": expected_workers,
            "frozen_worker_ids": frozen_worker_ids,
            "workers": [proof.model_dump(mode="json") for proof in ordered],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(**payload, proof_digest=digest)

    @model_validator(mode="after")
    def _validate_proof(self) -> "GenerationCutCoordinatorProof":
        worker_ids = [worker.worker_id for worker in self.workers]
        if tuple(sorted(self.frozen_worker_ids)) != self.frozen_worker_ids or len(self.frozen_worker_ids) != len(
            set(self.frozen_worker_ids)
        ):
            raise ValueError("coordinator frozen worker IDs must be unique and sorted")
        if len(self.frozen_worker_ids) != self.expected_workers:
            raise ValueError("coordinator frozen worker IDs do not match expected worker count")
        if worker_ids != sorted(worker_ids) or len(worker_ids) != len(set(worker_ids)):
            raise ValueError("coordinator generation-cut proof must contain unique sorted workers")
        if tuple(worker_ids) != self.frozen_worker_ids:
            raise ValueError("coordinator generation-cut proof does not match frozen worker IDs")
        if any(
            worker.checkpoint_id != self.checkpoint_id or worker.coordinator_sequence != self.coordinator_sequence
            for worker in self.workers
        ):
            raise ValueError("worker generation-cut proof has mismatched checkpoint or coordinator sequence")
        if any(worker.generation_pending for worker in self.workers):
            raise ValueError("coordinator generation-cut proof contains checkpoint-pending workers")
        payload = {
            "checkpoint_id": self.checkpoint_id,
            "coordinator_sequence": self.coordinator_sequence,
            "expected_workers": self.expected_workers,
            "frozen_worker_ids": self.frozen_worker_ids,
            "workers": [proof.model_dump(mode="json") for proof in self.workers],
        }
        expected_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.proof_digest != expected_digest:
            raise ValueError("coordinator generation-cut proof digest does not match")
        return self


class GenerationCutBackend(Protocol):
    """Framework-owned durable generation-cut lifecycle."""

    async def checkpoint_generation_cut(self, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        """Create durable recovery state and return its final receipt."""
        ...

    async def restore_generation_cut(self, receipt: GenerationCutReceipt) -> GenerationCutReceipt:
        """Restore the durable backend snapshot named by ``receipt``."""
        ...


class ModelAdmissionPauseRequest(CheckpointControlRequest):
    pass


class ModelAdmissionResumeRequest(CheckpointControlRequest):
    pass


class ModelAbortInflightRequest(CheckpointControlRequest):
    rollout_id: str = Field(min_length=1, pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
