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
"""Bulk checkpoint participant for stateful resources servers."""

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

from fastapi import FastAPI, Header, Query
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym._checkpoint.control import (
    CheckpointControlRequest,
    CheckpointPhase,
    ControlError,
    ControlFence,
    Deadline,
)
from nemo_gym.rollout_correlation import (
    ATTEMPT_INDEX_HEADER,
    ROLLOUT_ID_HEADER,
    ROLLOUT_ID_PATTERN,
    current_execution_identity,
)
from nemo_gym.token_id_capture.control_routes import require_control_auth


RESOURCES_CHECKPOINT_URL_PREFIX = "/ng-control/v1/resources-checkpoint"
RESOURCES_STATE_SUBDIR = "resources"
RESOURCES_MANIFEST_NAME = "manifest.json"
RESOURCES_CHECKPOINT_SCHEMA_VERSION = 2
RESOURCE_STATE_REVISION_HEADER = "x-nemo-gym-resource-state-revision"
EXPECTED_RESOURCE_STATE_REVISION_HEADER = "x-nemo-gym-expected-resource-state-revision"
RESOURCE_REQUEST_ID_HEADER = "x-nemo-gym-resource-request-id"

ResourcesRouteKind = Literal["read", "start", "mutation", "terminal"]
_CURRENT_TERMINAL_INTENT: ContextVar[bool] = ContextVar("nemo_gym_resource_terminal_intent", default=False)


class ResourcesCheckpointError(ControlError):
    code = "resources_checkpoint_error"


class ResourcesAdmissionClosedError(ControlError):
    code = "resources_admission_closed"


class ResourcesStaleAttemptError(ControlError):
    code = "stale_attempt"


class ResourcesSessionUnboundError(ControlError):
    code = "resources_session_unbound"


class ResourcesUnsafeRestoreError(ControlError):
    code = "unsafe_resources_restore"


class ResourceRevisionMismatchError(ControlError):
    code = "resource_revision_mismatch"


class ResourceRequestReceiptError(ControlError):
    code = "resource_request_receipt_error"


class ResourceReceiptCapacityError(ControlError):
    code = "resource_receipt_capacity_exhausted"


class ResourceReceiptEvictedError(ControlError):
    code = "resource_receipt_evicted"


class ResourcesUncertainMutationError(ControlError):
    code = "resource_mutation_uncertain"


class ResourceMutationReceipt(BaseModel):
    """Replayable response for one uniquely identified resource mutation."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=512)
    expected_revision: Optional[int] = Field(default=None, ge=0)
    input_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: int = Field(ge=100, le=599)
    headers: list[tuple[str, str]]
    body_base64: str
    resulting_revision: Optional[int] = Field(default=None, ge=0)

    def body(self) -> bytes:
        return base64.b64decode(self.body_base64)


class ResourceSnapshot(BaseModel):
    """One resources server's state after a completed tool mutation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, RESOURCES_CHECKPOINT_SCHEMA_VERSION] = RESOURCES_CHECKPOINT_SCHEMA_VERSION
    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)
    state_revision: int = Field(ge=0)
    state: dict[str, Any]
    mutation_receipts: list[ResourceMutationReceipt] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class ResourcesPrepareRequest(CheckpointControlRequest):
    pass


class ResourcesCommitRequest(CheckpointControlRequest):
    checkpoint_dir: str


class ResourcesRestoreRequest(CheckpointControlRequest):
    checkpoint_dir: str


class ResourcesResumeRequest(CheckpointControlRequest):
    pass


class ResourcesRetireRequest(CheckpointControlRequest):
    """Identify one execution to retire from an active checkpoint."""

    rollout_id: str = Field(pattern=ROLLOUT_ID_PATTERN.pattern)
    attempt_index: int = Field(ge=0)


class _LockEntry:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


class ResourcesCheckpointParticipant:
    """Serialize mutations per execution and export all known live sessions."""

    def __init__(
        self,
        *,
        export_state: Callable[[str, int], Awaitable[dict[str, Any]]],
        restore_states: Callable[[list[ResourceSnapshot]], Awaitable[None]],
        retire_state: Optional[Callable[[str, int], Awaitable[None]]] = None,
        restore_expected: bool = False,
        max_receipts_per_session: int = 1024,
        max_retired_receipt_sessions: int = 1024,
        max_evicted_receipt_tombstones: int = 4096,
    ) -> None:
        if max_receipts_per_session < 1 or max_retired_receipt_sessions < 0 or max_evicted_receipt_tombstones < 0:
            raise ValueError("resource receipt limits must be non-negative and per-session capacity must be positive")
        self._export_state = export_state
        self._restore_states = restore_states
        self._retire_state = retire_state
        self._locks: dict[tuple[str, int], _LockEntry] = {}
        self._revisions: dict[tuple[str, int], int] = {}
        self._receipts: dict[tuple[str, int], dict[str, ResourceMutationReceipt]] = {}
        self._retired_receipt_keys: deque[tuple[str, int]] = deque()
        self._max_receipts_per_session = max_receipts_per_session
        self._max_retired_receipt_sessions = max_retired_receipt_sessions
        self._max_evicted_receipt_tombstones = max_evicted_receipt_tombstones
        self._prepared: list[ResourceSnapshot] = []
        self._tombstones: set[tuple[str, int]] = set()
        self._evicted_receipt_tombstones: set[tuple[str, int]] = set()
        self._receipt_history_saturated = False
        self._uncertain: dict[tuple[str, int], str] = {}
        self._restore_expected = restore_expected
        self._served_stateful_traffic = False
        self._untracked_stateful_traffic = False
        self._accepting = not restore_expected
        self._prepare_lock = asyncio.Lock()

    @asynccontextmanager
    async def mutation_lock(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[None]:
        """Hold one session lock and prune it after the final user releases it."""
        key = (rollout_id, attempt_index)
        entry = self._locks.setdefault(key, _LockEntry())
        entry.users += 1
        acquired = False
        try:
            if timeout is None:
                await entry.lock.acquire()
            else:
                await asyncio.wait_for(entry.lock.acquire(), timeout=timeout)
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            entry.users -= 1
            self._prune_lock(key, entry)

    def _prune_lock(self, key: tuple[str, int], entry: Optional[_LockEntry] = None) -> None:
        entry = entry or self._locks.get(key)
        if (
            entry is not None
            and entry.users == 0
            and not entry.lock.locked()
            and key not in self._revisions
            and self._locks.get(key) is entry
        ):
            self._locks.pop(key, None)

    def bind(self, rollout_id: str, attempt_index: int) -> None:
        """Bind a successfully created logical session to its execution."""
        self._served_stateful_traffic = True
        self._revisions.setdefault((rollout_id, attempt_index), 0)

    def register(self, rollout_id: str, attempt_index: int) -> None:
        """Compatibility alias for tests and adapters that predate explicit binding."""
        self.bind(rollout_id, attempt_index)

    def is_bound(self, rollout_id: str, attempt_index: int) -> bool:
        return (rollout_id, attempt_index) in self._revisions

    def revision_for(self, rollout_id: str, attempt_index: int) -> Optional[int]:
        return self._revisions.get((rollout_id, attempt_index))

    def receipt_for(
        self,
        rollout_id: str,
        attempt_index: int,
        request_id: str,
    ) -> Optional[ResourceMutationReceipt]:
        return self._receipts.get((rollout_id, attempt_index), {}).get(request_id)

    def ensure_receipt_capacity(self, rollout_id: str, attempt_index: int, request_id: str) -> None:
        receipts = self._receipts.get((rollout_id, attempt_index), {})
        if request_id not in receipts and len(receipts) >= self._max_receipts_per_session:
            raise ResourceReceiptCapacityError(
                f"resource receipt capacity is exhausted for rollout {rollout_id!r} attempt {attempt_index}"
            )

    def record_receipt(
        self,
        rollout_id: str,
        attempt_index: int,
        receipt: ResourceMutationReceipt,
    ) -> None:
        self.ensure_receipt_capacity(rollout_id, attempt_index, receipt.request_id)
        self._receipts.setdefault((rollout_id, attempt_index), {})[receipt.request_id] = receipt

    def record_untracked_stateful_traffic(self) -> None:
        """Remember legacy traffic that cannot be included in a safe checkpoint."""
        self._served_stateful_traffic = True
        self._untracked_stateful_traffic = True

    def is_tombstoned(self, rollout_id: str, attempt_index: int) -> bool:
        return (rollout_id, attempt_index) in self._tombstones

    def mark_terminal_after_request(self, rollout_id: str, attempt_index: int) -> None:
        """Declare terminal intent for the current in-flight handler."""
        _CURRENT_TERMINAL_INTENT.set(True)

    def terminal_after_request(self, rollout_id: str, attempt_index: int) -> bool:
        return _CURRENT_TERMINAL_INTENT.get()

    def is_receipt_evicted(self, rollout_id: str, attempt_index: int) -> bool:
        return self._receipt_history_saturated or (rollout_id, attempt_index) in self._evicted_receipt_tombstones

    def _remember_evicted_receipts(self, key: tuple[str, int]) -> None:
        if self._receipt_history_saturated:
            return
        self._evicted_receipt_tombstones.add(key)
        if len(self._evicted_receipt_tombstones) > self._max_evicted_receipt_tombstones:
            self._evicted_receipt_tombstones.clear()
            self._receipt_history_saturated = True

    def uncertain_reason(self, rollout_id: str, attempt_index: int) -> Optional[str]:
        return self._uncertain.get((rollout_id, attempt_index))

    def mark_uncertain(self, rollout_id: str, attempt_index: int, reason: str) -> None:
        self._uncertain[(rollout_id, attempt_index)] = reason

    def complete_response(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        kind: ResourcesRouteKind,
        successful: bool,
        terminal: bool,
        resulting_revision: Optional[int],
        receipt: Optional[ResourceMutationReceipt],
    ) -> None:
        """Atomically publish metadata for one fully buffered response."""
        key = (rollout_id, attempt_index)
        if successful and kind == "start":
            self.bind(rollout_id, attempt_index)
        if resulting_revision is not None:
            if terminal:
                self._revisions.pop(key, None)
            else:
                self._revisions[key] = resulting_revision
        if receipt is not None:
            self.record_receipt(rollout_id, attempt_index, receipt)
        if terminal:
            self.retire(rollout_id, attempt_index, preserve_receipts=True)

    def retire(self, rollout_id: str, attempt_index: int, *, preserve_receipts: bool = False) -> None:
        key = (rollout_id, attempt_index)
        self._revisions.pop(key, None)
        self._uncertain.pop(key, None)
        if preserve_receipts and key in self._receipts:
            self._retired_receipt_keys.append(key)
            while len(self._retired_receipt_keys) > self._max_retired_receipt_sessions:
                retired_key = self._retired_receipt_keys.popleft()
                self._receipts.pop(retired_key, None)
                self._remember_evicted_receipts(retired_key)
                self._prune_lock(retired_key)
        else:
            self._receipts.pop(key, None)
            self._retired_receipt_keys = deque(
                retired_key for retired_key in self._retired_receipt_keys if retired_key != key
            )
        self._prune_lock(key)

    async def retire_execution(
        self,
        rollout_id: str,
        attempt_index: int,
        *,
        deadline_ts: Optional[float] = None,
    ) -> dict[str, Any]:
        """Fence and remove one sacrificed execution before checkpoint prepare."""
        key = (rollout_id, attempt_index)
        self._tombstones.add(key)
        timeout = None if deadline_ts is None else max(0.0, deadline_ts - time.time())
        try:
            async with self.mutation_lock(rollout_id, attempt_index, timeout=timeout):
                if self._retire_state is not None:
                    await self._retire_state(rollout_id, attempt_index)
                self.retire(rollout_id, attempt_index)
                self._prepared = [
                    snapshot for snapshot in self._prepared if (snapshot.rollout_id, snapshot.attempt_index) != key
                ]
        except asyncio.TimeoutError as error:
            raise ResourcesCheckpointError(
                f"timed out retiring resources state for rollout {rollout_id!r} attempt {attempt_index}"
            ) from error
        return {"retired": True, "rollout_id": rollout_id, "attempt_index": attempt_index}

    def record_mutation(self, rollout_id: str, attempt_index: int) -> int:
        key = (rollout_id, attempt_index)
        revision = self._revisions.get(key, 0) + 1
        self._revisions[key] = revision
        return revision

    @property
    def accepting(self) -> bool:
        return self._accepting

    async def prepare(self, deadline_ts: float) -> dict[str, Any]:
        async with self._prepare_lock:
            self._accepting = False
            if self._untracked_stateful_traffic:
                raise ResourcesCheckpointError(
                    "stateful traffic without execution identity was served; refusing to omit live state"
                )
            if self._uncertain:
                poisoned = [
                    {"rollout_id": key[0], "attempt_index": key[1], "reason": reason}
                    for key, reason in sorted(self._uncertain.items())
                ]
                raise ResourcesUncertainMutationError(
                    f"resource mutations have uncertain outcomes and must be retired before prepare: {poisoned}"
                )
            snapshots: list[ResourceSnapshot] = []
            for rollout_id, attempt_index in sorted(self._revisions):
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    raise ResourcesCheckpointError("deadline expired before all resources sessions were exported")
                try:
                    async with self.mutation_lock(rollout_id, attempt_index, timeout=remaining):
                        key = (rollout_id, attempt_index)
                        revision = self._revisions.get(key)
                        # A terminal request may retire the session while prepare waits for its lock.
                        # A retired session is no longer part of the cut.
                        if revision is None:
                            continue
                        state = await self._export_state(rollout_id, attempt_index)
                except asyncio.TimeoutError as error:
                    raise ResourcesCheckpointError(
                        f"timed out draining resources state for rollout {rollout_id!r} attempt {attempt_index}"
                    ) from error
                snapshots.append(
                    ResourceSnapshot(
                        rollout_id=rollout_id,
                        attempt_index=attempt_index,
                        state_revision=revision,
                        state=state,
                        mutation_receipts=list(self._receipts.get((rollout_id, attempt_index), {}).values()),
                    )
                )
            self._prepared = snapshots
            return {
                "sessions": len(snapshots),
                "state": "prepared",
                "inventory": self._snapshot_inventory(snapshots),
            }

    async def restore(self, snapshots: list[ResourceSnapshot]) -> None:
        self._accepting = False
        if self._served_stateful_traffic or self._revisions or self._receipts or self._uncertain:
            raise ResourcesUnsafeRestoreError(
                "resources restore requires a fresh process that has not served stateful traffic"
            )
        replacements = [
            snapshot.model_copy(update={"attempt_index": snapshot.attempt_index + 1}) for snapshot in snapshots
        ]
        for snapshot in replacements:
            request_ids = {receipt.request_id for receipt in snapshot.mutation_receipts}
            if (
                len(request_ids) != len(snapshot.mutation_receipts)
                or len(request_ids) > self._max_receipts_per_session
            ):
                raise ResourcesCheckpointError(
                    f"resource receipt inventory is invalid for rollout {snapshot.rollout_id!r} "
                    f"attempt {snapshot.attempt_index}"
                )
        # The environment validates and activates the complete replacement set
        # as one operation. A per-session restore loop could expose a mixed cut.
        await self._restore_states(replacements)
        self._tombstones.update((snapshot.rollout_id, snapshot.attempt_index) for snapshot in snapshots)
        for snapshot in replacements:
            key = (snapshot.rollout_id, snapshot.attempt_index)
            self._revisions[key] = snapshot.state_revision
            if snapshot.mutation_receipts:
                self._receipts[key] = {receipt.request_id: receipt for receipt in snapshot.mutation_receipts}
        self._restore_expected = False

    def resume(self) -> dict[str, Any]:
        if self._restore_expected:
            raise ResourcesCheckpointError("resources process was started restore-expected and has not restored state")
        self._accepting = True
        self._prepared = []
        return {"state": "accepting"}

    def prepared_snapshots(self) -> list[ResourceSnapshot]:
        if self._accepting:
            raise ResourcesCheckpointError("resources state must be prepared before commit")
        return list(self._prepared)

    def status(self) -> dict[str, Any]:
        admission_state = "accepting" if self._accepting else "paused"
        return {
            "state": admission_state,
            "admission_state": admission_state,
            "sessions": len(self._revisions),
            "per_session": [
                {
                    "rollout_id": rollout_id,
                    "attempt_index": attempt_index,
                    "revision": revision,
                    "locked": bool((entry := self._locks.get((rollout_id, attempt_index))) and entry.lock.locked()),
                }
                for (rollout_id, attempt_index), revision in sorted(self._revisions.items())
            ],
            "lock_entries": len(self._locks),
            "locked_sessions": sum(entry.lock.locked() for entry in self._locks.values()),
            "tombstones": len(self._tombstones),
            "evicted_receipt_tombstones": len(self._evicted_receipt_tombstones),
            "receipt_history_saturated": self._receipt_history_saturated,
            "uncertain_sessions": [
                {"rollout_id": key[0], "attempt_index": key[1], "reason": reason}
                for key, reason in sorted(self._uncertain.items())
            ],
            "mutation_receipts": sum(len(receipts) for receipts in self._receipts.values()),
            "retired_receipt_sessions": len(self._retired_receipt_keys),
            "prepared_inventory": self._snapshot_inventory(self._prepared),
        }

    @staticmethod
    def _snapshot_inventory(snapshots: list[ResourceSnapshot]) -> list[dict[str, Any]]:
        return [
            {
                "rollout_id": snapshot.rollout_id,
                "attempt_index": snapshot.attempt_index,
                "revision": snapshot.state_revision,
                "mutation_receipts": len(snapshot.mutation_receipts),
            }
            for snapshot in snapshots
        ]


class ResourcesSessionMiddleware:
    """Fence and serialize state mutations for one rollout attempt."""

    def __init__(
        self,
        app: Any,
        participant: ResourcesCheckpointParticipant,
        route_kind: Optional[Callable[[str, str], Optional[ResourcesRouteKind]]] = None,
    ) -> None:
        self._app = app
        self._participant = participant
        self._route_kind = route_kind or self._default_route_kind

    @staticmethod
    def _default_route_kind(path: str, method: str) -> Optional[ResourcesRouteKind]:
        if method != "POST":
            return None
        return "terminal" if path == "/verify" else "mutation"

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        kind = self._route_kind(path, scope.get("method", "GET"))
        if kind is None:
            await self._app(scope, receive, send)
            return
        try:
            identity = self._identity(scope)
        except ValueError:
            await self._reject_identity(send)
            return
        if identity is None:
            if not self._participant.accepting:
                await self._reject(send)
                return

            async def send_untracked(message: dict[str, Any]) -> None:
                if (
                    kind in {"start", "mutation"}
                    and message.get("type") == "http.response.start"
                    and int(message.get("status", 500)) < 400
                ):
                    self._participant.record_untracked_stateful_traffic()
                await send(message)

            await self._app(scope, receive, send_untracked)
            return
        rollout_id, attempt_index = identity
        if self._participant.is_tombstoned(rollout_id, attempt_index):
            await self._reject_stale(send, rollout_id, attempt_index)
            return
        if not self._participant.accepting:
            await self._reject(send)
            return
        if self._participant.is_receipt_evicted(rollout_id, attempt_index):
            await self._reject_receipt_evicted(send, rollout_id, attempt_index)
            return
        if (uncertain_reason := self._participant.uncertain_reason(rollout_id, attempt_index)) is not None:
            await self._reject_uncertain(send, uncertain_reason)
            return
        try:
            expected_revision = self._expected_revision(scope)
        except ValueError:
            await self._reject_revision(send, expected=None, actual=None, malformed=True)
            return
        try:
            request_id = self._request_id(scope)
        except ValueError:
            await self._reject_receipt(send, "invalid resource request ID header")
            return
        if kind != "read" and request_id is None:
            await self._reject_receipt(send, "checkpoint-managed resource mutations require a resource request ID")
            return
        request_messages, request_body = await self._buffer_request(receive)
        input_digest = self._input_digest(scope, request_body)

        async def replay_receive() -> dict[str, Any]:
            if request_messages:
                return request_messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        async with self._participant.mutation_lock(rollout_id, attempt_index):
            # Re-check after waiting: prepare may have closed admission while
            # this request waited behind an earlier mutation.
            if not self._participant.accepting:
                await self._reject(send)
                return
            if self._participant.is_tombstoned(rollout_id, attempt_index):
                await self._reject_stale(send, rollout_id, attempt_index)
                return
            if self._participant.is_receipt_evicted(rollout_id, attempt_index):
                await self._reject_receipt_evicted(send, rollout_id, attempt_index)
                return
            if (uncertain_reason := self._participant.uncertain_reason(rollout_id, attempt_index)) is not None:
                await self._reject_uncertain(send, uncertain_reason)
                return
            if request_id is not None:
                receipt = self._participant.receipt_for(rollout_id, attempt_index, request_id)
                if receipt is not None:
                    if receipt.expected_revision != expected_revision or receipt.input_digest != input_digest:
                        await self._reject_receipt(
                            send,
                            "resource request ID was reused with a different expected revision or input",
                        )
                        return
                    await self._replay_receipt(send, receipt)
                    return
            if kind != "start" and not self._participant.is_bound(rollout_id, attempt_index):
                await self._reject_unbound(send, rollout_id, attempt_index)
                return
            actual_revision = self._participant.revision_for(rollout_id, attempt_index)
            if expected_revision is not None and actual_revision != expected_revision:
                await self._reject_revision(send, expected=expected_revision, actual=actual_revision)
                return
            if request_id is not None:
                try:
                    self._participant.ensure_receipt_capacity(rollout_id, attempt_index, request_id)
                except ResourceReceiptCapacityError as error:
                    await self._reject_receipt(send, error.detail, code=error.code)
                    return
            response_messages: list[dict[str, Any]] = []

            async def capture_response(message: dict[str, Any]) -> None:
                response_messages.append(message)

            terminal_token = _CURRENT_TERMINAL_INTENT.set(False)
            try:
                await self._app(scope, replay_receive, capture_response)
                terminal_intent = _CURRENT_TERMINAL_INTENT.get()
            except BaseException as error:
                self._participant.mark_uncertain(
                    rollout_id,
                    attempt_index,
                    f"handler raised {type(error).__name__} before completing a replayable response",
                )
                raise
            finally:
                _CURRENT_TERMINAL_INTENT.reset(terminal_token)

            try:
                response_start = self._validate_complete_response(response_messages)
            except ValueError as error:
                reason = str(error)
                self._participant.mark_uncertain(rollout_id, attempt_index, reason)
                await self._reject_uncertain(send, reason)
                return

            status = int(response_start["status"])
            successful = status < 400
            terminal = successful and (kind == "terminal" or terminal_intent)
            resulting_revision = None
            if successful and kind != "read":
                resulting_revision = (actual_revision or 0) + 1
                headers = list(response_start.get("headers") or ())
                headers.append(
                    (
                        RESOURCE_STATE_REVISION_HEADER.encode("ascii"),
                        str(resulting_revision).encode("ascii"),
                    )
                )
                response_start["headers"] = headers
            receipt = None
            if request_id is not None:
                receipt = self._build_receipt(
                    request_id=request_id,
                    expected_revision=expected_revision,
                    input_digest=input_digest,
                    messages=response_messages,
                )
            self._participant.complete_response(
                rollout_id,
                attempt_index,
                kind=kind,
                successful=successful,
                terminal=terminal,
                resulting_revision=resulting_revision,
                receipt=receipt,
            )
            for message in response_messages:
                await send(message)

    @staticmethod
    def _identity(scope: dict[str, Any]) -> Optional[tuple[str, int]]:
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_rollout = headers.get(ROLLOUT_ID_HEADER.encode())
        raw_attempt = headers.get(ATTEMPT_INDEX_HEADER.encode())
        if raw_rollout is None and raw_attempt is None:
            rollout_id, attempt_index = current_execution_identity()
            if rollout_id is None and attempt_index is None:
                return None
            if rollout_id is None or attempt_index is None:
                raise ValueError("partial execution identity")
            return rollout_id, attempt_index
        if raw_rollout is None or raw_attempt is None:
            raise ValueError("partial execution identity")
        try:
            rollout_id = raw_rollout.decode("ascii")
            attempt_index = int(raw_attempt.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            raise ValueError("malformed execution identity") from None
        if not ROLLOUT_ID_PATTERN.fullmatch(rollout_id) or attempt_index < 0:
            raise ValueError("malformed execution identity")
        return rollout_id, attempt_index

    @staticmethod
    def _expected_revision(scope: dict[str, Any]) -> Optional[int]:
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_revision = headers.get(EXPECTED_RESOURCE_STATE_REVISION_HEADER.encode())
        if raw_revision is None:
            return None
        try:
            revision = int(raw_revision.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            raise ValueError("malformed expected resource revision") from None
        if revision < 0:
            raise ValueError("malformed expected resource revision")
        return revision

    @staticmethod
    def _request_id(scope: dict[str, Any]) -> Optional[str]:
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_request_id = headers.get(RESOURCE_REQUEST_ID_HEADER.encode())
        if raw_request_id is None:
            return None
        try:
            request_id = raw_request_id.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("malformed resource request ID") from None
        if (
            not request_id
            or len(request_id.encode("ascii")) > 512
            or any(ord(character) < 32 or ord(character) == 127 for character in request_id)
        ):
            raise ValueError("malformed resource request ID")
        return request_id

    @staticmethod
    async def _buffer_request(receive: Any) -> tuple[list[dict[str, Any]], bytes]:
        messages: list[dict[str, Any]] = []
        body_parts: list[bytes] = []
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        return messages, b"".join(body_parts)

    @staticmethod
    def _input_digest(scope: dict[str, Any], body: bytes) -> str:
        headers: dict[bytes, list[bytes]] = {}
        for key, value in scope.get("headers", ()):
            headers.setdefault(key.lower(), []).append(value)
        cookies = SimpleCookie()
        for raw_cookie in headers.get(b"cookie", []):
            cookies.load(raw_cookie.decode("latin-1"))
        content_types = sorted(
            ResourcesSessionMiddleware._canonical_content_type(value.decode("latin-1"))
            for value in headers.get(b"content-type", [])
        )
        semantic_headers = json.dumps(
            {
                "content-type": content_types,
                "cookies": sorted((name, morsel.value) for name, morsel in cookies.items()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256()
        digest.update(scope.get("method", "GET").encode("ascii"))
        digest.update(b"\0")
        digest.update(scope.get("path", "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(scope.get("query_string", b""))
        digest.update(b"\0")
        digest.update(semantic_headers)
        digest.update(b"\0")
        digest.update(body)
        return digest.hexdigest()

    @staticmethod
    def _canonical_content_type(value: str) -> str:
        media_type, *parameters = value.split(";")
        normalized_parameters = sorted(parameter.strip().lower() for parameter in parameters if parameter.strip())
        return ";".join([media_type.strip().lower(), *normalized_parameters])

    @staticmethod
    def _validate_complete_response(messages: list[dict[str, Any]]) -> dict[str, Any]:
        response_start: Optional[dict[str, Any]] = None
        body_started = False
        body_complete = False
        for message in messages:
            message_type = message.get("type")
            if message_type == "http.response.start":
                if response_start is not None or body_started:
                    raise ValueError("resource handler emitted an invalid duplicate or late response start")
                response_start = message
            elif message_type == "http.response.body":
                if response_start is None or body_complete:
                    raise ValueError("resource handler emitted an invalid response body sequence")
                body_started = True
                body_complete = not message.get("more_body", False)
            else:
                raise ValueError(f"resource handler emitted unsupported ASGI message {message_type!r}")
        if response_start is None or not body_started or not body_complete:
            raise ValueError("resource handler did not complete its ASGI response body")
        return response_start

    @staticmethod
    def _build_receipt(
        *,
        request_id: str,
        expected_revision: Optional[int],
        input_digest: str,
        messages: list[dict[str, Any]],
    ) -> ResourceMutationReceipt:
        start = next(message for message in messages if message.get("type") == "http.response.start")
        headers = [(key.decode("latin-1"), value.decode("latin-1")) for key, value in start.get("headers", ())]
        revision = next(
            (int(value) for key, value in headers if key.lower() == RESOURCE_STATE_REVISION_HEADER),
            None,
        )
        body = b"".join(
            message.get("body", b"") for message in messages if message.get("type") == "http.response.body"
        )
        return ResourceMutationReceipt(
            request_id=request_id,
            expected_revision=expected_revision,
            input_digest=input_digest,
            status=int(start["status"]),
            headers=headers,
            body_base64=base64.b64encode(body).decode("ascii"),
            resulting_revision=revision,
        )

    @staticmethod
    async def _replay_receipt(send: Any, receipt: ResourceMutationReceipt) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": receipt.status,
                "headers": [(key.encode("latin-1"), value.encode("latin-1")) for key, value in receipt.headers],
            }
        )
        await send({"type": "http.response.body", "body": receipt.body()})

    @staticmethod
    async def _reject_receipt(
        send: Any,
        message: str,
        *,
        code: str = ResourceRequestReceiptError.code,
    ) -> None:
        payload = json.dumps({"error": {"code": code, "message": message}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_receipt_evicted(send: Any, rollout_id: str, attempt_index: int) -> None:
        await ResourcesSessionMiddleware._reject_receipt(
            send,
            f"resource receipts were evicted for rollout {rollout_id!r} attempt {attempt_index}; "
            "the execution cannot be replayed",
            code=ResourceReceiptEvictedError.code,
        )

    @staticmethod
    async def _reject_uncertain(send: Any, reason: str) -> None:
        await ResourcesSessionMiddleware._reject_receipt(
            send,
            f"resource mutation outcome is uncertain: {reason}",
            code=ResourcesUncertainMutationError.code,
        )

    @staticmethod
    async def _reject(send: Any) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": ResourcesAdmissionClosedError.code,
                    "message": "resources mutation admission is closed for checkpoint preparation",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_identity(send: Any) -> None:
        payload = b'{"error":{"code":"execution_identity_mismatch","message":"invalid execution identity headers"}}'
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_unbound(send: Any, rollout_id: str, attempt_index: int) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": ResourcesSessionUnboundError.code,
                    "message": f"rollout {rollout_id!r} attempt {attempt_index} has no successful seed/reset binding",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_stale(send: Any, rollout_id: str, attempt_index: int) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": ResourcesStaleAttemptError.code,
                    "message": f"rollout {rollout_id!r} attempt {attempt_index} was retired by restore",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _reject_revision(
        send: Any,
        *,
        expected: Optional[int],
        actual: Optional[int],
        malformed: bool = False,
    ) -> None:
        message = (
            "invalid expected resource revision header"
            if malformed
            else f"expected resource revision {expected}, but the current revision is {actual}"
        )
        payload = json.dumps(
            {
                "error": {
                    "code": ResourceRevisionMismatchError.code,
                    "message": message,
                    "expected_revision": expected,
                    "actual_revision": actual,
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_resources_state(
    participant: ResourcesCheckpointParticipant,
    checkpoint_dir: Path,
    *,
    checkpoint_id: str,
    server_name: str,
) -> dict[str, Any]:
    directory = Path(checkpoint_dir) / RESOURCES_STATE_SUBDIR / server_name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / RESOURCES_MANIFEST_NAME
    if manifest_path.exists():
        return _validate_resources_manifest(directory, checkpoint_id=checkpoint_id, server_name=server_name)
    files: dict[str, str] = {}
    for snapshot in participant.prepared_snapshots():
        name = f"{snapshot.rollout_id}.a{snapshot.attempt_index}.json"
        path = directory / name
        _write_atomic(path, snapshot.model_dump_json(indent=2).encode())
        files[name] = _digest(path)
    _fsync_dir(directory)
    manifest = {
        "schema_version": RESOURCES_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "server_name": server_name,
        "files": files,
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
    _write_atomic(manifest_path, payload)
    _fsync_dir(directory)
    return {"sessions": len(files), "manifest_digest": hashlib.sha256(payload).hexdigest()}


def _validate_resources_manifest(directory: Path, *, checkpoint_id: str, server_name: str) -> dict[str, Any]:
    manifest_path = directory / RESOURCES_MANIFEST_NAME
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("checkpoint_id") != checkpoint_id or manifest.get("server_name") != server_name:
        raise ResourcesCheckpointError("resources checkpoint manifest belongs to a different transaction or server")
    for name, digest in manifest.get("files", {}).items():
        path = directory / name
        if not path.exists() or _digest(path) != digest:
            raise ResourcesCheckpointError(f"resources checkpoint state {name!r} is missing or corrupted")
    return {
        "sessions": len(manifest.get("files", {})),
        "manifest_digest": hashlib.sha256(payload).hexdigest(),
    }


def load_resources_state(checkpoint_dir: Path, *, server_name: str) -> tuple[str, list[ResourceSnapshot]]:
    directory = Path(checkpoint_dir) / RESOURCES_STATE_SUBDIR / server_name
    manifest_path = directory / RESOURCES_MANIFEST_NAME
    if not manifest_path.exists():
        raise ResourcesCheckpointError(f"resources checkpoint has no committed manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    snapshots: list[ResourceSnapshot] = []
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.exists() or _digest(path) != digest:
            raise ResourcesCheckpointError(f"resources checkpoint state {name!r} is missing or corrupted")
        snapshots.append(ResourceSnapshot.model_validate_json(path.read_bytes()))
    _reconcile_agent_resource_revisions(checkpoint_dir, server_name=server_name, snapshots=snapshots)
    return manifest["checkpoint_id"], snapshots


def _reconcile_agent_resource_revisions(
    checkpoint_dir: Path,
    *,
    server_name: str,
    snapshots: list[ResourceSnapshot],
) -> None:
    """Require exact agent/resources revisions when agent state is present."""
    agent_root = Path(checkpoint_dir) / "agent"
    if not agent_root.exists():
        return
    selected: dict[tuple[str, int], set[int]] = {}
    for manifest_path in agent_root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest.get("files", {}).items():
            record_path = manifest_path.parent / name
            if not record_path.exists() or _digest(record_path) != digest:
                raise ResourcesCheckpointError(f"agent checkpoint record {name!r} is missing or corrupted")
            record = json.loads(record_path.read_text())
            revisions = record.get("resource_state_revisions") or {}
            if server_name not in revisions:
                continue
            key = (record["rollout_id"], int(record["attempt_index"]))
            selected.setdefault(key, set()).add(int(revisions[server_name]))

    actual = {(snapshot.rollout_id, snapshot.attempt_index): snapshot.state_revision for snapshot in snapshots}
    expected = {key: revisions for key, revisions in selected.items()}
    if set(actual) != set(expected):
        raise ResourceRevisionMismatchError(
            f"resources checkpoint inventory {sorted(actual)} does not match selected agent boundaries {sorted(expected)}"
        )
    for key, revision in actual.items():
        if expected[key] != {revision}:
            raise ResourceRevisionMismatchError(
                f"resources revision {revision} for rollout {key[0]!r} attempt {key[1]} "
                f"does not match selected agent boundary revisions {sorted(expected[key])}"
            )


def install_resources_checkpoint(
    app: FastAPI,
    *,
    participant: ResourcesCheckpointParticipant,
    fence: ControlFence,
    auth_token: str,
    server_name: str,
    route_kind: Callable[[str, str], Optional[ResourcesRouteKind]],
) -> None:
    """Install bulk prepare, commit, restore, and resume routes."""

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/prepare")
    async def prepare(
        body: ResourcesPrepareRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await participant.prepare(body.deadline_ts)

        result = await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/prepare",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.PREPARING,
            phase_after=CheckpointPhase.PREPARED,
            run=run,
            deadline=body,
        )
        return result

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/commit")
    async def commit(
        body: ResourcesCommitRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return await asyncio.to_thread(
                commit_resources_state,
                participant,
                Path(body.checkpoint_dir),
                checkpoint_id=body.checkpoint_id,
                server_name=server_name,
            )

        return await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/commit",
            allowed_phases=frozenset({CheckpointPhase.PREPARED}),
            phase_during=CheckpointPhase.COMMITTING,
            phase_after=CheckpointPhase.COMMITTED_PAUSED,
            run=run,
        )

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/restore")
    async def restore(
        body: ResourcesRestoreRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            source_checkpoint_id, snapshots = await asyncio.to_thread(
                load_resources_state,
                Path(body.checkpoint_dir),
                server_name=server_name,
            )
            await participant.restore(snapshots)
            return {"sessions": len(snapshots), "source_checkpoint_id": source_checkpoint_id}

        return await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/restore",
            allowed_phases=frozenset({CheckpointPhase.IDLE}),
            phase_during=CheckpointPhase.RESTORING,
            phase_after=CheckpointPhase.RESTORED_PAUSED,
            run=run,
        )

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/resume")
    async def resume(
        body: ResourcesResumeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)

        async def run() -> dict[str, Any]:
            return participant.resume()

        return await fence.run_operation(
            body.checkpoint_id,
            "resources-checkpoint/resume",
            allowed_phases=frozenset(
                {
                    CheckpointPhase.IDLE,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
            phase_during=fence.phase,
            phase_after=CheckpointPhase.IDLE,
            run=run,
            retire_outcome="resumed",
        )

    @app.post(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/retire")
    async def retire(
        body: ResourcesRetireRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        fence.require_phase(
            body.checkpoint_id,
            frozenset({CheckpointPhase.PREPARING, CheckpointPhase.PREPARED}),
        )
        return await participant.retire_execution(
            body.rollout_id,
            body.attempt_index,
            deadline_ts=body.deadline_ts,
        )

    @app.get(f"{RESOURCES_CHECKPOINT_URL_PREFIX}/status")
    async def status(
        checkpoint_id: str = Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
        deadline_ts: float = Query(),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, auth_token)
        fence.require_phase(
            checkpoint_id,
            frozenset(
                {
                    CheckpointPhase.PREPARING,
                    CheckpointPhase.PREPARED,
                    CheckpointPhase.COMMITTING,
                    CheckpointPhase.COMMITTED_PAUSED,
                    CheckpointPhase.RESTORING,
                    CheckpointPhase.RESTORED_PAUSED,
                }
            ),
        )
        Deadline(deadline_ts=deadline_ts)
        return {"checkpoint_id": checkpoint_id, **participant.status()}

    app.add_middleware(ResourcesSessionMiddleware, participant=participant, route_kind=route_kind)
