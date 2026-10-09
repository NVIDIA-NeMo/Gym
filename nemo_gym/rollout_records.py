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
"""Index saved rollout outcomes for collection and offline aggregation.

Results and failure sidecars are the source of truth. Execution numbers are
reserved in the run manifest; no dispatch or outcome event journal is written.
"""

import os
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

import orjson
from pydantic import ValidationError

from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_outcomes import RolloutFailure
from nemo_gym.rollout_recovery import (
    RunManifest,
    _digest,
    _get_max_rollout_attempts,
    is_terminal_failure,
    manifest_path_for,
    observed_elapsed,
)


RUN_ID_KEY = "_ng_run_id"


def journal_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_attempts.jsonl")


def coverage_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_coverage.json")


def materialized_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_materialized_inputs.jsonl")


def resolve_rollout_path(output: Path, *, read_only: bool = False) -> Path:
    """Resolve an output without losing its companions.

    Readers may retain an unambiguous legacy alias layout. Writers require all
    recovery files beside the resolved output, so aliases cannot split a run.
    Derived coverage reports do not establish ownership of a run.
    """
    resolved = output.resolve()
    companions = (manifest_path_for, journal_path_for, materialized_path_for, failures_path_for)
    misplaced = [
        companion(output)
        for companion in companions
        if companion(output).resolve() != companion(resolved).resolve()
        and (companion(output).exists() or companion(output).is_symlink())
    ]
    if not misplaced:
        return resolved
    if read_only and not any(
        companion(path).exists() or companion(path).is_symlink()
        for path in (output, resolved)
        for companion in (manifest_path_for, journal_path_for)
    ):
        # Do not mix or silently discard two competing sets of legacy files.
        if not any(companion(resolved).exists() or companion(resolved).is_symlink() for companion in companions):
            return output.absolute()
    raise ConfigError(
        f"Recovery artifacts exist beside the rollout alias ({misplaced[0]}). "
        "Use the original Gym revision, or move all companions beside the resolved output "
        f"({resolved}) after backing up the run. Saved artifacts were not changed."
    )


def logical_rollout_id(row: dict) -> str:
    logical = {key: value for key, value in row.items() if key != ATTEMPT_INDEX_KEY_NAME}
    try:
        identity = maybe_rollout_id_from_run_body(logical)
        if identity is not None:
            EpisodeId(rollout_id=identity)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"Invalid rollout identity: {error}") from error
    if identity is None:
        raise ConfigError("Recovery requires a rollout id or materialized task/rollout indices.")
    return identity


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """A persisted JSONL record, loaded only when its contents are requested.

    Legacy migration may assign an attempt index without rewriting the source.
    Offsets and lengths are bytes, including any final newline.
    """

    path: Path
    offset: int
    length: int
    line_number: int = 0
    legacy_attempt_index: int | None = None
    file_identity: tuple[int, int] | None = None
    expected_identity: tuple[str | None, str, int] | None = None

    def read(self) -> dict:
        """Read this record with its effective legacy attempt identity."""
        with self.path.open("rb") as file:
            return self._read(file)

    def _read(self, file: BinaryIO) -> dict:
        file.seek(self.offset)
        row = orjson.loads(file.read(self.length))
        if self.legacy_attempt_index is not None:
            row[ATTEMPT_INDEX_KEY_NAME] = self.legacy_attempt_index
        if self.expected_identity is not None:
            actual = (row.get(RUN_ID_KEY), logical_rollout_id(row), row.get(ATTEMPT_INDEX_KEY_NAME, 0))
            if actual != self.expected_identity:
                raise ConfigError(f"Saved record identity changed after indexing {self.path} at byte {self.offset}.")
        return row


@dataclass(frozen=True, slots=True)
class _Outcome:
    record: RolloutRecord
    failure_class: str | None
    terminal: bool
    masked: bool
    has_reward: bool
    reuse_cached_deliverable: bool
    elapsed_seconds: float | None


def _indexed_records(path: Path) -> Iterator[tuple[RolloutRecord, dict]]:
    """Scan one record at a time, retaining its original byte coordinates."""
    if not path.exists():
        return
    with path.open("rb") as file:
        stat = os.fstat(file.fileno())
        file_identity = (stat.st_dev, stat.st_ino)
        offset = 0
        for number, raw in enumerate(file, 1):
            record = RolloutRecord(path, offset, len(raw), number, file_identity=file_identity)
            offset += len(raw)
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
            yield record, value


def read_records(path: Path) -> Iterator[dict]:
    """Read committed JSONL records; an incomplete final write is not a record."""
    for _, row in _indexed_records(path):
        yield row


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
        # Search each chunk once. Growing and rescanning a bytes buffer here is
        # quadratic for large interrupted trajectory records.
        while start:
            size = min(start, 1 << 20)
            start -= size
            file.seek(start)
            split = file.read(size).rfind(b"\n")
            if split >= 0:
                start += split + 1
                break
        file.seek(start)
        tail = file.read(end - start)
        try:
            orjson.loads(tail)
        except orjson.JSONDecodeError:
            file.truncate(start)
            warnings.warn(f"Removed {end - start} bytes of incomplete final JSON from {path}.", stacklevel=2)
        else:
            file.seek(0, 2)
            file.write(b"\n")
        file.flush()


def _is_invalid_judge_migration(original: dict, migrated: dict) -> bool:
    """Recognize only the exact sidecar-first migration already supported by Gym."""
    if not (
        original.get("invalid_judge_response")
        and original.get("_ng_failure_class") is None
        and migrated.get("_ng_migrated_invalid_judge_response") is True
        and migrated.get("_ng_failure_class") == "judge_invalid"
        and "_ng_failure_terminal" not in migrated
    ):
        return False
    changed = {
        "_ng_failure_class",
        "_ng_failure_terminal",
        "reuse_cached_deliverable",
        "_ng_migrated_invalid_judge_response",
    }
    return {k: v for k, v in original.items() if k not in changed} == {
        k: v for k, v in migrated.items() if k not in changed
    }


class RolloutRecords:
    def __init__(self, manifest: RunManifest, expected: list[dict]):
        self.manifest = manifest
        self.expected = {}
        for row in expected:
            identity = logical_rollout_id(row)
            if identity in self.expected:
                raise ConfigError(f"Duplicate logical rollout id {identity!r} in materialized inputs.")
            self.expected[identity] = row
        self.attempt_counts: Counter = Counter()
        self.overridden_terminal_counts: Counter = Counter()
        self.latest: dict[str, int] = {
            identity: index - 1 for identity, index in manifest.next_attempt.items() if index > 0
        }
        self.payloads: dict[tuple[str, int], _Outcome] = {}
        self.omitted: set[tuple[str, int]] = set()
        self.retry_terminal_timeouts = False

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

    def check_run_identity(self, row: dict, *, legacy: bool = False) -> None:
        if row.get(RUN_ID_KEY) != self.manifest.run_id and not (legacy and RUN_ID_KEY not in row):
            raise ConfigError("Saved outcome belongs to a different run.")

    def check_outcome(self, row: dict, *, legacy: bool = False) -> tuple[str, int]:
        """Validate without mutation, before a writer appends the payload."""
        key = self._key(row)
        self.check_run_identity(row, legacy=legacy)
        if "_ng_failure_record" in row:
            try:
                failure = RolloutFailure.model_validate(row["_ng_failure_record"])
            except ValidationError as error:
                raise ConfigError(
                    f"Invalid saved failure record for rollout attempt {key!r}: {error}. "
                    "Upgrade Gym or explicitly migrate the record before resuming."
                ) from error
            if (
                failure.run_id != self.manifest.run_id
                or (failure.episode_id.rollout_id, failure.episode_id.attempt) != key
                or row.get("_ng_failure_class") != failure.sidecar_failure_class
                or bool(row.get("_ng_failure_terminal")) != failure.failure.terminal
            ):
                raise ConfigError(
                    f"Saved failure record is inconsistent with its envelope for rollout attempt {key!r}."
                )
        if not legacy and key[1] >= self.manifest.next_attempt.get(key[0], 0):
            raise ConfigError(f"Saved outcome {key!r} has no reserved attempt in the run manifest.")
        previous = self.payloads.get(key)
        if previous is not None and previous.record.read() != row:
            raise ConfigError(f"Conflicting outcomes for rollout attempt {key!r}.")
        return key

    def _payload(self, row: dict, record: RolloutRecord, *, legacy: bool = False) -> None:
        key = self.check_outcome(row, legacy=legacy)
        self.latest[key[0]] = max(key[1], self.latest.get(key[0], -1))
        failure = row.get("_ng_failure_record")
        delivery = failure.get("delivery") if isinstance(failure, dict) else "possibly_delivered"
        if (
            key not in self.payloads
            and row.get("_ng_failure_class") is not None
            and delivery != "not_sent"
            and not row.get("_ng_failure_terminal")
        ):
            self.attempt_counts[key[0]] += 1
        if (
            key not in self.payloads
            and row.get("_ng_failure_class") is not None
            and delivery != "not_sent"
            and row.get("_ng_failure_terminal")
            and not is_terminal_failure(row, retry_terminal_timeouts=True)
        ):
            self.overridden_terminal_counts[key[0]] += 1
        if row.get("_ng_omitted"):
            self.omitted.add(key)
        self.payloads[key] = _Outcome(
            record=replace(record, expected_identity=(row.get(RUN_ID_KEY), key[0], key[1])),
            failure_class=row.get("_ng_failure_class"),
            terminal=bool(row.get("_ng_failure_terminal")),
            masked=bool(row.get("mask_sample")),
            has_reward=type(row.get("reward")) in (int, float),
            reuse_cached_deliverable=bool(row.get("reuse_cached_deliverable")),
            elapsed_seconds=observed_elapsed(row),
        )

    def outcome(self, row: dict, *, record: RolloutRecord) -> None:
        """Index an outcome after its JSONL record has been flushed."""
        self._payload(row, record)

    @classmethod
    def load(cls, output: Path, manifest: RunManifest, *, import_legacy: bool = False) -> "RolloutRecords":
        expected = list(read_records(materialized_path_for(output)))
        if _digest(expected) != manifest.materialized_digest:
            raise ConfigError("Saved materialized inputs do not match the run manifest.")
        state = cls(manifest, expected)
        unexpected = manifest.next_attempt.keys() - state.expected.keys()
        if unexpected:
            raise ConfigError(f"Run manifest reserves attempts for unknown rollouts: {sorted(unexpected)!r}.")
        # Legacy files lacked an attempt id on some rows. Import once in their
        # recorded order, preserving the old failure-count-based numbering.
        legacy_counts: Counter = Counter()
        for path in (failures_path_for(output), output):
            for record, payload in _indexed_records(path):
                if import_legacy and RUN_ID_KEY not in payload:
                    identity = logical_rollout_id(payload)
                    payload = dict(payload)
                    payload.setdefault(ATTEMPT_INDEX_KEY_NAME, legacy_counts[identity])
                    key = state._key(payload)
                    if key in state.payloads and state.payloads[key].record.read() != payload:
                        # Old append/reverify writers reused explicit attempt IDs.
                        # Only untagged legacy rows use arrival-order migration;
                        # manifest-backed records still reject conflicting payloads.
                        payload[ATTEMPT_INDEX_KEY_NAME] = max(legacy_counts[identity], key[1] + 1)
                    legacy_counts[identity] = max(legacy_counts[identity], payload[ATTEMPT_INDEX_KEY_NAME] + 1)
                    record = replace(record, legacy_attempt_index=payload[ATTEMPT_INDEX_KEY_NAME])
                if (path == output) == (payload.get("_ng_failure_class") is not None) and not (
                    import_legacy and RUN_ID_KEY not in payload
                ):
                    raise ConfigError(f"Outcome in the wrong artifact: {path}.")
                previous = state.payloads.get(state._key(payload))
                if path == output and previous is not None:
                    migrated = previous.record.read()
                    if _is_invalid_judge_migration(payload, migrated):
                        # A crash after the sidecar fsync but before main-file
                        # replacement leaves both copies. Only this exact,
                        # explicitly marked reclassification may supersede one.
                        state.check_run_identity(payload, legacy=import_legacy)
                        continue
                state._payload(payload, record, legacy=import_legacy and RUN_ID_KEY not in payload)
        return state

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
        if payload.failure_class == "skipped" and payload.terminal:
            return "omitted"
        return "failure" if payload.failure_class is not None else "success"

    def selected_records(self, disposition: str) -> dict[tuple[str, int], RolloutRecord]:
        """Select byte locations using attempt metadata, without loading payloads."""
        return {
            (identity, self.latest[identity]): self.payloads[(identity, self.latest[identity])].record
            for identity in self.expected
            if self.disposition(identity) == disposition and (identity, self.latest.get(identity)) in self.payloads
        }

    def selected(self, disposition: str) -> list[dict]:
        # Existing consumers explicitly request materialized results. Keep reads
        # out of reconciliation/coverage and reuse handles for random access.
        with ExitStack() as files:
            handles = {}
            rows = []
            for record in self.selected_records(disposition).values():
                if record.path not in handles:
                    handles[record.path] = files.enter_context(record.path.open("rb"))
                rows.append(record._read(handles[record.path]))
            return rows

    def _retryable(self, identity: str) -> bool:
        payload = self.payloads.get((identity, self.latest.get(identity)))
        terminal = payload.terminal if payload else False
        if payload and self.retry_terminal_timeouts:
            terminal = is_terminal_failure(
                {"_ng_failure_class": payload.failure_class, "_ng_failure_terminal": payload.terminal},
                retry_terminal_timeouts=True,
            )
        return self.disposition(identity) not in {"success", "omitted"} and not terminal

    def failure_count(self, identity: str) -> int:
        return self.attempt_counts[identity] + (
            self.overridden_terminal_counts[identity] if self.retry_terminal_timeouts else 0
        )

    def exhausted_count(self, max_attempts: int) -> int:
        return sum(
            self._retryable(identity) and self.failure_count(identity) >= max_attempts for identity in self.expected
        )

    def pending(self, max_attempts: int) -> list[dict]:
        pending = []
        for identity, original in self.expected.items():
            if not self._retryable(identity) or self.failure_count(identity) >= max_attempts:
                continue
            row = dict(original)
            row[ATTEMPT_INDEX_KEY_NAME] = self.manifest.next_attempt.get(identity, self.latest.get(identity, -1) + 1)
            pending.append(row)
        return pending

    def coverage(self) -> dict:
        counts = Counter(self.disposition(identity) for identity in self.expected)
        expected = len(self.expected)
        max_attempts = _get_max_rollout_attempts()
        # A producer-masked result completed execution, so recovery still reuses
        # it. Report its measurement status separately, after selecting attempts.
        masked = sum(
            self.payloads[(identity, self.latest[identity])].masked
            and self.payloads[(identity, self.latest[identity])].has_reward
            for identity in self.expected
            if self.disposition(identity) == "success"
        )
        unscored = sum(
            not self.payloads[(identity, self.latest[identity])].has_reward
            for identity in self.expected
            if self.disposition(identity) == "success"
        )
        return {
            "schema_version": 2,
            "selection_policy": self.manifest.selection_policy,
            "run_id": self.manifest.run_id,
            "expected": expected,
            "successful": counts["success"],
            "measured": counts["success"] - masked - unscored,
            "masked": masked,
            "unscored": unscored,
            "failed": counts["failure"],
            "intentionally_omitted": counts["omitted"],
            "unknown": counts["unknown"],
            "never_dispatched": expected - len(self.latest),
            "attempts": sum(
                max(self.manifest.next_attempt.get(identity, 0), index + 1) for identity, index in self.latest.items()
            ),
            "counted_failures": sum(self.failure_count(identity) for identity in self.expected),
            "attempts_exhausted": self.exhausted_count(max_attempts),
            "retryable": sum(
                self._retryable(identity) and self.failure_count(identity) < max_attempts for identity in self.expected
            ),
            "max_rollout_attempts": max_attempts,
            "retry_terminal_timeouts": self.retry_terminal_timeouts,
            "completion_fraction": counts["success"] / expected if expected else 1.0,
            "complete": counts["success"] == expected,
            "reconciled": counts["unknown"] == 0,
            "identity_verified": not (self.manifest.legacy_import or self.manifest.identity_overridden),
        }
