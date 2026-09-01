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
"""Shared control-plane infrastructure for checkpoint coordination.

Every Gym server exposes ``GET /ng-control/v1/capabilities`` so the NeMo-RL
NemoGym actor can discover, before any checkpoint starts, what each server
supports: its component kind, admission states, checkpoint mode, schema
version, concurrency contract, and multi-process mode. A required capability
that is missing fails setup on the actor side instead of failing the first
checkpoint at its deadline.

Control routes added by later features (admission pause/resume, checkpoint
commit/restore) share one fencing discipline, implemented here by
``ControlFence``:

- Every control call carries a ``checkpoint_id`` and an absolute deadline.
- A retried call with the same checkpoint id and operation returns the first
  call's recorded result instead of running the operation again.
- A call whose checkpoint id was already committed or aborted is rejected
  with ``409 stale_checkpoint``; a call for a different checkpoint while one
  is active is rejected with ``409 checkpoint_conflict``.
- An operation arriving in a phase it is not legal in is rejected with
  ``409 invalid_phase``.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


CONTROL_URL_PREFIX = "/ng-control/v1"
CONTROL_SCHEMA_VERSION = 1


class AdmissionState(str, Enum):
    """Data-plane admission states a server can be in.

    ``ACCEPTING`` grants capacity normally. ``DRAINING`` refuses new
    top-level operations while accepted ones finish. ``PAUSED`` means the
    drain completed: nothing is in flight and nothing new is admitted.
    """

    ACCEPTING = "accepting"
    DRAINING = "draining"
    PAUSED = "paused"


class CheckpointPhase(str, Enum):
    """Server-local phase of checkpoint participation.

    ``IDLE`` is normal operation. Prepare moves through ``PREPARING`` to
    ``PREPARED`` (admission closed, drain complete). Commit moves through
    ``COMMITTING`` to ``COMMITTED_PAUSED`` (state exported, still closed).
    Restore on a fresh process moves through ``RESTORING`` to
    ``RESTORED_PAUSED``. An explicit resume or abort returns the server to
    ``IDLE`` and retires the checkpoint id.
    """

    IDLE = "idle"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED_PAUSED = "committed_paused"
    RESTORING = "restoring"
    RESTORED_PAUSED = "restored_paused"


class ControlError(Exception):
    """A control-plane rejection with a stable machine-readable code."""

    status_code = 409
    code = "control_error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class StaleCheckpointError(ControlError):
    code = "stale_checkpoint"


class CheckpointConflictError(ControlError):
    code = "checkpoint_conflict"


class InvalidPhaseError(ControlError):
    code = "invalid_phase"


class Deadline(BaseModel):
    """An absolute wall-clock deadline shared across machines.

    A relative timeout would restart at each participant.
    An absolute deadline keeps every participant inside the controller's budget.
    """

    model_config = ConfigDict(extra="forbid")

    deadline_ts: FiniteFloat = Field(description="Absolute deadline as seconds since the Unix epoch.")

    def remaining(self, now: Optional[float] = None) -> float:
        return max(0.0, self.deadline_ts - (time.time() if now is None else now))

    def expired(self, now: Optional[float] = None) -> bool:
        return self.remaining(now) == 0.0


class CheckpointControlRequest(Deadline):
    """Fields shared by checkpoint control requests."""

    schema_version: Literal[CONTROL_SCHEMA_VERSION] = CONTROL_SCHEMA_VERSION
    checkpoint_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MultiProcessCapability(BaseModel):
    """How this server keeps checkpoint state consistent across processes.

    ``single_worker`` is a validated one-process deployment: in-process state
    is service state. ``coordinator`` means a service-level coordinator owns
    the control URL and aggregates worker acknowledgements. ``unmanaged``
    means the server runs multiple workers without a coordinator; the actor
    must refuse to checkpoint through it.
    """

    mode: Literal["single_worker", "coordinator", "unmanaged"]
    num_workers: int = 1


class ControlCapabilities(BaseModel):
    """The capability declaration served at ``GET /ng-control/v1/capabilities``."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    component: Literal["responses_api_models", "responses_api_agents", "resources_servers"]
    name: str
    schema_version: int = CONTROL_SCHEMA_VERSION
    admission_states: list[AdmissionState] = Field(
        default_factory=lambda: [AdmissionState.ACCEPTING],
        description="Admission states this server can enter. A server without an admission "
        "limiter only ever accepts; the actor must not ask it to pause.",
    )
    checkpoint_mode: Literal["stateless", "export_restore"] = Field(
        default="stateless",
        description="'stateless' means the server has nothing to export: its rollouts restore "
        "as fresh dispatches. 'export_restore' means it can export and restore per-rollout state.",
    )
    concurrency_contract: Literal["stateless", "serialized_per_session", "transactional_parallel"] = "stateless"
    multi_process: MultiProcessCapability


def multi_process_capability_from_num_workers(num_workers: Optional[int]) -> MultiProcessCapability:
    """Derive the multi-process declaration from a configured worker count.

    One worker (or an unset count, which runs one worker) is the validated
    ``single_worker`` mode. More than one worker without a coordinator is
    declared ``unmanaged`` so the actor refuses it instead of reading one
    arbitrary worker's in-process state as service state.
    """
    workers = num_workers or 1
    if workers <= 1:
        return MultiProcessCapability(mode="single_worker", num_workers=1)
    return MultiProcessCapability(mode="unmanaged", num_workers=workers)


class ControlFence:
    """Checkpoint-id fencing and phase machine for one server process.

    One fence guards all control routes of a server. Operations run through
    ``run_operation``, which provides idempotent replay, duplicate-call
    coalescing, stale-id rejection, cross-checkpoint conflict rejection, and
    phase validation. State transitions commit only when the operation
    succeeds; a failed operation restores the entry phase so the coordinator
    can retry or abort.
    """

    def __init__(self) -> None:
        self.phase = CheckpointPhase.IDLE
        self.active_checkpoint_id: Optional[str] = None
        self.deadline: Optional[Deadline] = None
        self._results: dict[tuple[str, str], dict[str, Any]] = {}
        self._inflight: dict[tuple[str, str], asyncio.Future] = {}
        self._retired: dict[str, str] = {}
        self._retired_results: dict[tuple[str, str], dict[str, Any]] = {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "active_checkpoint_id": self.active_checkpoint_id,
            "deadline_ts": self.deadline.deadline_ts if self.deadline is not None else None,
        }

    def _validate(self, checkpoint_id: str, allowed_phases: frozenset[CheckpointPhase]) -> None:
        if checkpoint_id in self._retired:
            raise StaleCheckpointError(
                f"checkpoint {checkpoint_id!r} already finished with outcome "
                f"{self._retired[checkpoint_id]!r}; this call is from a stale coordinator"
            )
        if self.active_checkpoint_id is not None and self.active_checkpoint_id != checkpoint_id:
            raise CheckpointConflictError(
                f"checkpoint {self.active_checkpoint_id!r} is active in phase {self.phase.value!r}; "
                f"refusing operation for {checkpoint_id!r}"
            )
        if self.phase not in allowed_phases:
            raise InvalidPhaseError(
                f"operation not valid in phase {self.phase.value!r} "
                f"(valid from: {sorted(phase.value for phase in allowed_phases)})"
            )

    async def run_operation(
        self,
        checkpoint_id: str,
        operation: str,
        *,
        allowed_phases: frozenset[CheckpointPhase],
        phase_during: CheckpointPhase,
        phase_after: CheckpointPhase,
        run: Callable[[], Awaitable[dict[str, Any]]],
        deadline: Optional[Deadline] = None,
        retire_outcome: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run one fenced control operation and record its result.

        A repeat call with the same ``(checkpoint_id, operation)`` replays the
        recorded result; a concurrent duplicate awaits the in-flight run
        instead of starting a second one. ``retire_outcome`` marks the
        checkpoint finished after this operation (resume or abort): the fence
        returns to ``IDLE`` and the id becomes stale forever.
        """
        key = (checkpoint_id, operation)
        if checkpoint_id in self._retired:
            final_result = self._retired_results.get(key)
            if final_result is not None:
                return final_result
            self._validate(checkpoint_id, allowed_phases)
        recorded = self._results.get(key)
        if recorded is not None:
            return recorded
        inflight = self._inflight.get(key)
        if inflight is not None:
            return await asyncio.shield(inflight)

        self._validate(checkpoint_id, allowed_phases)

        entry_phase = self.phase
        entry_deadline = self.deadline
        self.active_checkpoint_id = checkpoint_id
        self.phase = phase_during
        if deadline is not None:
            self.deadline = deadline
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            result = await run()
        except BaseException as e:
            self.phase = entry_phase
            self.deadline = entry_deadline
            if entry_phase == CheckpointPhase.IDLE:
                self.active_checkpoint_id = None
            future.set_exception(e)
            # A coalesced duplicate re-raises through the shielded await;
            # nothing may be left awaiting silently.
            if not future.cancelled():
                future.exception()
            raise
        finally:
            self._inflight.pop(key, None)

        self.phase = phase_after
        self._results[key] = result
        if retire_outcome is not None:
            self._retire(checkpoint_id, retire_outcome, final_key=key)
        future.set_result(result)
        return result

    def _retire(self, checkpoint_id: str, outcome: str, *, final_key: tuple[str, str]) -> None:
        self._retired[checkpoint_id] = outcome
        self._retired_results[final_key] = self._results[final_key]
        self._results = {key: value for key, value in self._results.items() if key[0] != checkpoint_id}
        self.active_checkpoint_id = None
        self.deadline = None
        self.phase = CheckpointPhase.IDLE


def install_control_plane(app: FastAPI, *, capabilities: ControlCapabilities, fence: ControlFence) -> None:
    """Register the shared ``/ng-control/v1`` routes on a server app.

    The capabilities route stays reachable while the data plane is paused:
    control routes are plain FastAPI routes, not gated by any admission
    limiter, so the coordinator can always observe a draining server.
    """

    @app.get(f"{CONTROL_URL_PREFIX}/capabilities")
    async def control_capabilities() -> dict[str, Any]:
        return {**capabilities.model_dump(), **fence.snapshot()}

    async def control_error_handler(request: Any, exc: ControlError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "detail": exc.detail}},
        )

    app.add_exception_handler(ControlError, control_error_handler)
