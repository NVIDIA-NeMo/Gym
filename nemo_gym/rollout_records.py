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

import hashlib
import math
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

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.core.episode import EpisodeId, maybe_rollout_id_from_run_body
from nemo_gym.rollout_outcomes import RolloutFailureRecord
from nemo_gym.rollout_recovery import (
    RunManifest,
    _digest,
    _get_max_rollout_attempts,
    has_saved_generation,
    has_unclassified_execution_failure,
    is_terminal_failure,
    load_resubmit_offset,
    manifest_path_for,
    matches_failure_class,
    observed_elapsed,
)
from nemo_gym.utils.path_utils import failures_path_for, resubmit_offset_path_for


RUN_ID_KEY = "_ng_run_id"
# Names the saved judge failure a judge-only re-judgment settles; the result keeps that failure's attempt identity.
REJUDGED_FAILURE_ID_KEY = "_ng_rejudged_failure_id"
# The pending_rejudge operation an outcome closes.
REJUDGE_OPERATION_ID_KEY = "_ng_rejudge_operation_id"
# The settled judge failure's own observations, kept on the selected result.
REJUDGED_OBSERVATIONS_KEY = "_ng_rejudged_failure_observations"


def journal_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_attempts.jsonl")


def coverage_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_coverage.json")


def materialized_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_materialized_inputs.jsonl")


def resolve_rollout_path(output: Path, *, read_only: bool = False) -> Path:
    """Resolve an output without losing its companions.

    Readers may retain an unambiguous legacy alias layout. Recovery writers
    require all history beside the resolved output, so aliases cannot split a run.
    Legacy-only reverification may reuse reader resolution after excluding history.
    Derived coverage reports do not establish ownership of a run.
    """
    resolved = output.resolve()
    companions = (
        manifest_path_for,
        journal_path_for,
        materialized_path_for,
        failures_path_for,
        resubmit_offset_path_for,
    )
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
            return output.parent.resolve() / output.name
    raise ConfigError(
        f"Recovery artifacts exist beside the rollout alias ({misplaced[0]}). "
        "Use the Gym revision that created these files, or move all companions beside the resolved output "
        f"({resolved}) after backing up the run. Saved artifacts were not changed."
    )


def resolve_rollout_owner(path: Path) -> Path:
    """Find a run's main output when given its failure sidecar or saved inputs.

    An output with its own history owns that history regardless of its name;
    strip the suffix only when the parent run has a manifest or journal.
    """
    output = resolve_rollout_path(path, read_only=True)
    if not any(companion(output).exists() for companion in (manifest_path_for, journal_path_for)):
        for suffix in ("_failures", "_materialized_inputs"):
            if output.stem.endswith(suffix):
                parent = output.with_name(output.stem.removesuffix(suffix) + output.suffix)
                parent = resolve_rollout_path(parent, read_only=True)
                if any(companion(parent).exists() for companion in (manifest_path_for, journal_path_for)):
                    return parent
    return output


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
    legacy_failure_class: str | None
    terminal: bool
    masked: bool
    has_reward: bool
    elapsed_seconds: float | None
    final_zero: bool = False
    interruption_cause_unknown: bool = False
    saved_answer: bool = False
    unclassified_execution_failure: bool = False


def _indexed_records(path: Path, *, end_offset: int | None = None) -> Iterator[tuple[RolloutRecord, dict]]:
    """Scan one record at a time, retaining its original byte coordinates."""
    if not path.exists():
        return
    with path.open("rb") as file:
        stat = os.fstat(file.fileno())
        file_identity = (stat.st_dev, stat.st_ino)
        offset = 0
        number = 0
        while end_offset is None or offset < end_offset:
            raw = file.readline(-1 if end_offset is None else end_offset - offset)
            if not raw:
                break
            number += 1
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


def is_interruption_failure(row: dict) -> bool:
    """Identify failures that spend neither the failure budget nor retry progress."""
    failure = row.get("_ng_failure_record")
    return matches_failure_class(row, {"kill_shaped"}) or (
        isinstance(failure, dict)
        and (failure.get("delivery") == "not_sent" or failure.get("failure", {}).get("stage") == "admission")
    )


def _holds_saved_answer(row: dict) -> bool:
    """Use the same retained-generation facts as collection and recovery."""
    return has_saved_generation(row)


def rejudgment_admission_closed_by(pending, outcome: dict, identity: str, failures_path: Path) -> bool:
    """Whether outcome is the saved result of admission pending: operation, predecessor id and row hash, identity,
    attempt and admission time all match. Cold readers and the close use this one check."""
    if (
        outcome.get(REJUDGE_OPERATION_ID_KEY) != pending.operation_id
        or outcome.get(REJUDGED_FAILURE_ID_KEY) != pending.predecessor_failure_id
        or outcome.get("_ng_rejudge_admitted_at") != pending.admitted_at
    ):
        return False
    for _, row in _indexed_records(failures_path):
        if row.get("_ng_failure_id") == pending.predecessor_failure_id:
            return (
                logical_rollout_id(row) == identity
                and row.get(ATTEMPT_INDEX_KEY_NAME, 0) == outcome.get(ATTEMPT_INDEX_KEY_NAME, 0)
                and hashlib.sha256(orjson.dumps(row, option=orjson.OPT_SORT_KEYS)).hexdigest()
                == pending.predecessor_sha256
            )
    return False


def is_rejudgment_of(failure: dict, result: dict) -> bool:
    """Whether ``result`` re-scores the saved answer of judge failure ``failure`` without a new model attempt."""
    return (
        failure.get("_ng_failure_class") == "judge_failed"
        and _holds_saved_answer(failure)
        and failure.get("_ng_failure_id") is not None
        and result.get("_ng_failure_class") is None
        and type(result.get("reward")) in (int, float)
        and math.isfinite(result["reward"])
        and result.get("evaluation_completed") is not False
        and not result.get("mask_sample")
        and result.get(RUN_ID_KEY) == failure.get(RUN_ID_KEY)
        and result.get(ATTEMPT_INDEX_KEY_NAME, 0) == failure.get(ATTEMPT_INDEX_KEY_NAME, 0)
        and result.get(REJUDGED_FAILURE_ID_KEY) == failure["_ng_failure_id"]
        and result.get("response") == failure.get("response")
    )


def is_rejudgment_failure_of(failure: dict, row: dict) -> bool:
    """Whether ``row`` records that re-judging the saved answer of judge failure ``failure`` failed again."""
    return (
        failure.get("_ng_failure_class") == "judge_failed"
        and _holds_saved_answer(failure)
        and failure.get("_ng_failure_id") is not None
        and row.get("_ng_failure_class") == "judge_failed"
        and row.get("_ng_failure_id") not in (None, failure["_ng_failure_id"])
        and row.get(RUN_ID_KEY) == failure.get(RUN_ID_KEY)
        and row.get(ATTEMPT_INDEX_KEY_NAME, 0) == failure.get(ATTEMPT_INDEX_KEY_NAME, 0)
        and row.get(REJUDGED_FAILURE_ID_KEY) == failure["_ng_failure_id"]
        and row.get("response") == failure.get("response")
    )


def _has_eligible_interruption_cause(row: dict) -> bool:
    """Whether the saved interruption names a cause resume may act on: a request never sent or refused at admission."""
    failure = row.get("_ng_failure_record")
    return (
        isinstance(failure, dict)
        and not _holds_saved_answer(row)
        and (failure.get("failure") or {}).get("stage") != "verification"
        and (failure.get("delivery") == "not_sent" or (failure.get("failure") or {}).get("stage") == "admission")
    )


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
        # An unanswered reservation cannot erase an earlier outcome's stop decision.
        self.replay_blocked: set[str] = set()
        self.unresolved_dispatches: set[str] = set()
        self.retry_terminal_timeouts = manifest.retry_terminal_timeouts

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
                failure = RolloutFailureRecord.model_validate(row["_ng_failure_record"])
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
        if previous is not None:
            saved = previous.record.read()
            if saved != row and not (
                is_rejudgment_of(saved, row) or is_rejudgment_of(row, saved) or is_rejudgment_failure_of(saved, row)
            ):
                raise ConfigError(f"Conflicting outcomes for rollout attempt {key!r}.")
        return key

    def _payload(self, row: dict, record: RolloutRecord, *, legacy: bool = False, count_failure: bool = True) -> None:
        key = self.check_outcome(row, legacy=legacy)
        previous = self.payloads.get(key)
        if previous is not None and previous.failure_class is None and row.get("_ng_failure_class") is not None:
            # The judge failure that a re-judgment of the same attempt already settled: its debit stays and the
            # settled result stays selected.
            if count_failure and not row.get("_ng_failure_terminal"):
                self.attempt_counts[key[0]] += 1
            self.replay_blocked.add(key[0])
            return
        if previous is not None and count_failure and is_rejudgment_failure_of(previous.record.read(), row):
            # A failed re-judgment spends the same failure budget as the judge failure it re-judged; the model
            # attempt and its reservation stay as they were.
            self.attempt_counts[key[0]] += 1
        self.latest[key[0]] = max(key[1], self.latest.get(key[0], -1))
        interruption = is_interruption_failure(row)
        if (
            count_failure
            and key not in self.payloads
            and row.get("_ng_failure_class") is not None
            and not interruption
            and not row.get("_ng_failure_terminal")
        ):
            self.attempt_counts[key[0]] += 1
        if (
            count_failure
            and key not in self.payloads
            and row.get("_ng_failure_class") is not None
            and not interruption
            and row.get("_ng_failure_terminal")
            and not is_terminal_failure(row, retry_terminal_timeouts=True)
        ):
            self.overridden_terminal_counts[key[0]] += 1
        if row.get("_ng_omitted"):
            self.omitted.add(key)
        self.payloads[key] = _Outcome(
            record=replace(record, expected_identity=(row.get(RUN_ID_KEY), key[0], key[1])),
            failure_class=row.get("_ng_failure_class"),
            legacy_failure_class=row.get("_ng_failure_legacy_class"),
            terminal=bool(row.get("_ng_failure_terminal")),
            masked=bool(row.get("mask_sample")),
            has_reward=type(row.get("reward")) in (int, float),
            elapsed_seconds=observed_elapsed(row),
            # The producer settled this rollout as a scored zero: final, so no policy reopens it.
            final_zero=row.get("_ng_failure_counts_as_zero") is True and bool(row.get("_ng_failure_terminal")),
            interruption_cause_unknown=interruption and not _has_eligible_interruption_cause(row),
            unclassified_execution_failure=has_unclassified_execution_failure(row),
            saved_answer=row.get("_ng_failure_class") is not None and _holds_saved_answer(row),
        )
        outcome = self.payloads[key]
        if (
            outcome.saved_answer
            or outcome.final_zero
            or outcome.interruption_cause_unknown
            or outcome.unclassified_execution_failure
        ):
            self.replay_blocked.add(key[0])

    def outcome(self, row: dict, *, record: RolloutRecord) -> None:
        """Index an outcome after its JSONL record has been flushed."""
        self._payload(row, record)

    @classmethod
    def load(
        cls,
        output: Path,
        manifest: RunManifest,
        *,
        import_legacy: bool = False,
        outcome_limits: dict[Path, int] | None = None,
    ) -> "RolloutRecords":
        """Index outcomes, optionally restricted to prefixes captured before the manifest read."""
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
        resubmit_offset = load_resubmit_offset(output)
        resubmitted = resubmit_offset_path_for(output).exists()
        for path in (failures_path_for(output), output):
            for record, payload in _indexed_records(
                path, end_offset=outcome_limits[path] if outcome_limits is not None else None
            ):
                if payload.get("_ng_omitted_rollout") and (
                    payload.get("_ng_dispatch_drained") or payload.get("_ng_unrouted_run_error")
                ):
                    # Keep diagnostics out of outcome selection and scoring. An unresolved current dispatch,
                    # however, cannot be replayed merely because its outcome was lost with the response.
                    if payload.get("_ng_unrouted_run_error") and has_unclassified_execution_failure(payload):
                        key = state._key(payload)
                        state.check_run_identity(payload, legacy=import_legacy)
                        if key[1] == manifest.next_attempt.get(key[0], 0) - 1:
                            state.unresolved_dispatches.add(key[0])
                    continue
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
                state._payload(
                    payload,
                    record,
                    legacy=import_legacy and RUN_ID_KEY not in payload,
                    # Legacy imports can contain failures in the main file.
                    # All of those precede the reset: managed writers append
                    # subsequent failures exclusively to the sidecar.
                    count_failure=(record.line_number > resubmit_offset)
                    if path == failures_path_for(output)
                    else not resubmitted,
                )
        for identity, pending in manifest.pending_rejudge.items():
            latest = state.payloads.get((identity, state.latest.get(identity)))
            if latest is None or not rejudgment_admission_closed_by(
                pending, latest.record.read(), identity, failures_path_for(output)
            ):
                # Admitted for re-judgment but no outcome saved: whether the judge ran is unknown.
                state.unresolved_dispatches.add(identity)
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
        if payload and payload.final_zero:
            return False
        if identity in self.unresolved_dispatches:
            # The current dispatch is unresolved even if an older outcome still supplies score/history.
            return False
        if payload is None and identity in self.replay_blocked:
            # A missing newer outcome does not reopen a retained answer, settled zero or unknown interruption.
            return False
        if payload and payload.unclassified_execution_failure:
            # A route failure is evidence of an attempt, not permission to repeat an unknown execution.
            return False
        if payload and payload.saved_answer:
            # A finished answer is re-judged by reverification, never regenerated by running the rollout again.
            return False
        if payload and payload.interruption_cause_unknown:
            # An interruption with no recorded eligible cause stops automatic recovery; its history and the
            # failures counted before it stay, and a saved answer is never regenerated.
            return False
        if payload and self.retry_terminal_timeouts:
            terminal = is_terminal_failure(
                {
                    "_ng_failure_class": payload.failure_class,
                    "_ng_failure_legacy_class": payload.legacy_failure_class,
                    "_ng_failure_terminal": payload.terminal,
                },
                retry_terminal_timeouts=True,
            )
        return self.disposition(identity) not in {"success", "omitted"} and not terminal

    def failure_count(self, identity: str) -> int:
        return self.attempt_counts[identity] + (
            self.overridden_terminal_counts[identity] if self.retry_terminal_timeouts else 0
        )

    def exhausted_count(self, max_attempts: int) -> int:
        exhausted = 0
        for identity in self.expected:
            outcome = self.payloads.get((identity, self.latest.get(identity)))
            # Stopping an unclassified execution does not refund the failures already spent at its cap.
            stopped_unknown = (
                outcome is not None
                and outcome.unclassified_execution_failure
                and not (
                    outcome.terminal
                    or outcome.final_zero
                    or outcome.saved_answer
                    or outcome.interruption_cause_unknown
                )
            )
            exhausted += (self._retryable(identity) or stopped_unknown) and self.failure_count(
                identity
            ) >= max_attempts
        return exhausted

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
        max_attempts = self.manifest.max_rollout_attempts or _get_max_rollout_attempts()
        not_started = expected - len(self.latest)
        missing = counts["unknown"] - not_started
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
        failures = self.selected_records("failure")
        # Settled final zeros are complete and in the denominator, and still failures for health.
        final_zeros = sum(self.payloads[key].final_zero for key in failures)
        failed_final = sum(
            not self._retryable(identity) or self.failure_count(identity) >= max_attempts for identity, _ in failures
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
            "failed_final": failed_final,
            "final_zeros": final_zeros,
            "failed_not_final": len(failures) - failed_final,
            "failed_by_class": dict(sorted(Counter(self.payloads[key].failure_class for key in failures).items())),
            "intentionally_omitted": counts["omitted"],
            "not_started": not_started,
            "missing": missing,
            "attempts": sum(
                max(self.manifest.next_attempt.get(identity, 0), index + 1) for identity, index in self.latest.items()
            ),
            "counted_failures": sum(self.failure_count(identity) for identity in self.expected),
            "failure_budget_exhausted": self.exhausted_count(max_attempts),
            "retryable": sum(
                self._retryable(identity) and self.failure_count(identity) < max_attempts for identity in self.expected
            ),
            "max_rollout_attempts": max_attempts,
            "retry_terminal_timeouts": self.retry_terminal_timeouts,
            "completion_fraction": (counts["success"] + final_zeros) / expected if expected else 1.0,
            "complete": counts["success"] + final_zeros == expected,
            "reconciled": missing == 0,
            "identity_verified": not (self.manifest.legacy_import or self.manifest.identity_overridden),
        }
