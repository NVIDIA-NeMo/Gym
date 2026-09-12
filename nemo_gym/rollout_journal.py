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
"""Single-writer attempt history and reconciliation for Gym's evaluation runner.

Dispatch is flushed before the request starts. Payloads stay in the existing
result/sidecar artifacts, carrying the manifest's run id. A complete payload can
therefore be recovered even if the collector died before journaling its outcome.
The greatest dispatched attempt index wins, independent of arrival order.
"""

import warnings
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_recovery import RunManifest, _digest


RUN_ID_KEY = "_ng_run_id"


def journal_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_attempts.jsonl")


def coverage_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_coverage.json")


def materialized_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_materialized_inputs.jsonl")


def logical_rollout_id(row: dict) -> str:
    logical = {key: value for key, value in row.items() if key != ATTEMPT_INDEX_KEY_NAME}
    try:
        identity = maybe_rollout_id_from_run_body(logical)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"Invalid rollout identity: {error}") from error
    if identity is None:
        raise ConfigError("Recovery requires a rollout id or materialized task/rollout indices.")
    return identity


def read_records(path: Path):
    """Read committed JSONL records; an incomplete final write is not a record."""
    if not path.exists():
        return
    with path.open("rb") as file:
        for number, raw in enumerate(file, 1):
            if not raw.strip():
                continue
            try:
                value = orjson.loads(raw)
            except orjson.JSONDecodeError as error:
                if not raw.endswith(b"\n"):
                    warnings.warn(f"Ignoring incomplete final record in {path} at line {number}.", stacklevel=2)
                    return
                raise ConfigError(f"Malformed JSON in {path} at line {number}: {error}") from error
            if not isinstance(value, dict):
                raise ConfigError(f"Expected an object in {path} at line {number}.")
            yield value


def prepare_append(path: Path) -> None:
    """Repair only an unterminated tail; preserve every complete history record."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r+b") as file:
        file.seek(-1, 2)
        if file.read(1) == b"\n":
            return
        end = file.tell()
        start = end
        tail = b""
        while start:
            size = min(start, 8192)
            start -= size
            file.seek(start)
            tail = file.read(size) + tail
            split = tail.rfind(b"\n")
            if split >= 0:
                start += split + 1
                tail = tail[split + 1 :]
                break
        try:
            orjson.loads(tail)
        except orjson.JSONDecodeError:
            file.truncate(start)
            warnings.warn(f"Removed {end - start} bytes of incomplete final JSON from {path}.", stacklevel=2)
        else:
            file.seek(0, 2)
            file.write(b"\n")
        file.flush()


class AttemptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str
    rollout_id: str
    attempt_index: int = Field(ge=0, strict=True)
    status: Literal["dispatched", "success", "failure", "omitted"]
    reason: str | None = Field(default=None, max_length=2000)


class RolloutJournal:
    def __init__(self, manifest: RunManifest, expected: list[dict]):
        self.manifest = manifest
        self.expected = {}
        for row in expected:
            identity = logical_rollout_id(row)
            if identity in self.expected:
                raise ConfigError(f"Duplicate logical rollout id {identity!r} in materialized inputs.")
            self.expected[identity] = row
        self.dispatched: set[tuple[str, int]] = set()
        self.latest: dict[str, int] = {}
        self.payloads: dict[tuple[str, int], dict] = {}
        self.omitted: set[tuple[str, int]] = set()
        self.file: BinaryIO | None = None

    def _key(self, row: dict) -> tuple[str, int]:
        identity = logical_rollout_id(row)
        index = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if identity not in self.expected:
            raise ConfigError(f"Rollout {identity!r} is not in this run's materialized inputs.")
        if type(index) is not int or index < 0:
            raise ConfigError("Attempt indices must be non-negative integers.")
        expected = self.expected[identity]
        for field in (TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME):
            if row.get(field) != expected.get(field):
                raise ConfigError(f"Rollout {identity!r} has mismatched {field}.")
        return identity, index

    def _dispatch(self, key: tuple[str, int]) -> None:
        self.dispatched.add(key)
        self.latest[key[0]] = max(key[1], self.latest.get(key[0], -1))

    def _event(self, key: tuple[str, int], status: str, reason: str | None = None) -> None:
        if self.file is None:
            raise RuntimeError("Attempt history is not open for writing.")
        event = AttemptEvent(
            run_id=self.manifest.run_id,
            rollout_id=key[0],
            attempt_index=key[1],
            status=status,
            reason=reason[:2000] if reason else None,
        )
        self.file.write(event.model_dump_json().encode() + b"\n")
        self.file.flush()

    def dispatch(self, row: dict) -> None:
        key = self._key(row)
        if key not in self.dispatched:
            self._event(key, "dispatched")
            self._dispatch(key)

    def check_outcome(self, row: dict, *, legacy: bool = False) -> tuple[str, int]:
        """Validate without mutation, before a writer appends the payload."""
        key = self._key(row)
        if row.get(RUN_ID_KEY) != self.manifest.run_id and not (legacy and RUN_ID_KEY not in row):
            raise ConfigError("Saved outcome belongs to a different run.")
        if not legacy and key not in self.dispatched:
            raise ConfigError(f"Saved outcome {key!r} has no dispatch in this run's attempt history.")
        previous = self.payloads.get(key)
        if previous is not None and previous != row:
            raise ConfigError(f"Conflicting outcomes for rollout attempt {key!r}.")
        if key in self.omitted:
            raise ConfigError(f"Omitted rollout attempt {key!r} also has an outcome.")
        return key

    def _payload(self, row: dict, *, legacy: bool = False) -> None:
        key = self.check_outcome(row, legacy=legacy)
        if legacy:
            self._dispatch(key)
        self.payloads[key] = row

    def outcome(self, row: dict) -> None:
        """Called after the payload artifact is flushed, so it is already recoverable."""
        key = self._key(row)
        self._payload(row)
        self._event(key, "failure" if row.get("_ng_failure_class") is not None else "success")

    def omit(self, row: dict, reason: str) -> None:
        self.dispatch(row)
        key = self._key(row)
        if key in self.payloads:
            raise ConfigError(f"Completed rollout attempt {key!r} cannot be omitted.")
        self._event(key, "omitted", reason)
        self.omitted.add(key)

    @classmethod
    def load(
        cls, output: Path, manifest: RunManifest, *, import_legacy: bool = False, rebuild_history: bool = False
    ) -> "RolloutJournal":
        expected = list(read_records(materialized_path_for(output)))
        if _digest(expected) != manifest.materialized_digest:
            raise ConfigError("Saved materialized inputs do not match the run manifest.")
        state = cls(manifest, expected)
        history = journal_path_for(output)
        if not history.exists() and not (import_legacy or rebuild_history):
            raise ConfigError(f"Cannot resume without attempt history: {history}.")
        for value in read_records(history):
            try:
                event = AttemptEvent.model_validate(value)
            except ValidationError as error:
                raise ConfigError(f"Invalid attempt history in {history}: {error}") from error
            if event.run_id != manifest.run_id or event.rollout_id not in state.expected:
                raise ConfigError("Attempt history belongs to a different run or input inventory.")
            key = event.rollout_id, event.attempt_index
            if event.status == "dispatched":
                state._dispatch(key)
            elif key not in state.dispatched:
                raise ConfigError(f"Outcome event {key!r} has no preceding dispatch.")
            elif event.status == "omitted":
                state.omitted.add(key)

        # Legacy files lacked an attempt id on some rows. Import once in their
        # recorded order, preserving the old failure-count-based numbering.
        legacy_counts: Counter = Counter()
        for path in (failures_path_for(output), output):
            for payload in read_records(path):
                if import_legacy and RUN_ID_KEY not in payload:
                    identity = logical_rollout_id(payload)
                    payload = dict(payload)
                    payload.setdefault(ATTEMPT_INDEX_KEY_NAME, legacy_counts[identity])
                    key = state._key(payload)
                    if key in state.payloads and state.payloads[key] != payload:
                        # Old append/reverify writers reused explicit attempt IDs.
                        # Only untagged legacy rows use arrival-order migration;
                        # journal-backed records still reject conflicting payloads.
                        payload[ATTEMPT_INDEX_KEY_NAME] = max(legacy_counts[identity], key[1] + 1)
                    legacy_counts[identity] = max(legacy_counts[identity], payload[ATTEMPT_INDEX_KEY_NAME] + 1)
                if (path == output) == (payload.get("_ng_failure_class") is not None) and not (
                    import_legacy and RUN_ID_KEY not in payload
                ):
                    raise ConfigError(f"Outcome in the wrong artifact: {path}.")
                state._payload(payload, legacy=rebuild_history or (import_legacy and RUN_ID_KEY not in payload))
        return state

    def seed_legacy_history(self) -> None:
        """Import existing outcomes explicitly; all subsequent writes have run identity."""
        for key, payload in self.payloads.items():
            self._event(key, "dispatched")
            self._event(key, "failure" if payload.get("_ng_failure_class") is not None else "success")

    def disposition(self, identity: str) -> str:
        index = self.latest.get(identity)
        if index is None:
            return "unknown"
        key = identity, index
        if key in self.omitted:
            return "omitted"
        payload = self.payloads.get(key)
        if payload is None:
            return "unknown"
        if payload.get("_ng_failure_class") == "skipped" and payload.get("_ng_failure_terminal"):
            return "omitted"
        return "failure" if payload.get("_ng_failure_class") is not None else "success"

    def selected(self, disposition: str) -> list[dict]:
        return [
            self.payloads[(identity, self.latest[identity])]
            for identity in self.expected
            if self.disposition(identity) == disposition and (identity, self.latest.get(identity)) in self.payloads
        ]

    def pending(self, max_attempts: int) -> list[dict]:
        attempts = Counter(identity for identity, _ in self.dispatched)
        pending = []
        for identity, original in self.expected.items():
            disposition = self.disposition(identity)
            payload = self.payloads.get((identity, self.latest.get(identity)), {})
            if disposition in {"success", "omitted"} or payload.get("_ng_failure_terminal"):
                continue
            if attempts[identity] >= max_attempts:
                continue
            row = dict(original)
            if identity in self.latest:
                row[ATTEMPT_INDEX_KEY_NAME] = self.latest[identity] + 1
            pending.append(row)
        return pending

    def coverage(self) -> dict:
        counts = Counter(self.disposition(identity) for identity in self.expected)
        expected = len(self.expected)
        # A producer-masked result completed execution, so recovery still reuses
        # it. Report its measurement status separately, after selecting attempts.
        masked = sum(bool(row.get("mask_sample")) for row in self.selected("success"))
        return {
            "schema_version": 1,
            "selection_policy": self.manifest.selection_policy,
            "run_id": self.manifest.run_id,
            "expected": expected,
            "successful": counts["success"],
            "measured": counts["success"] - masked,
            "masked": masked,
            "failed": counts["failure"],
            "intentionally_omitted": counts["omitted"],
            "unknown": counts["unknown"],
            "never_dispatched": expected - len(self.latest),
            "attempts": len(self.dispatched),
            "completion_fraction": counts["success"] / expected if expected else 1.0,
            "complete": counts["success"] == expected,
            "reconciled": counts["unknown"] == 0,
            "identity_verified": not (self.manifest.legacy_import or self.manifest.identity_overridden),
        }
