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
import re
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
from nemo_gym.core.episode_context import maybe_rollout_id_from_run_body
from nemo_gym.core.episode_types import EpisodeId
from nemo_gym.core.failure_kinds import (
    AGENT_TIMEOUT_INFRASTRUCTURE,
    ROLLOUT_ALLOCATION_ENDED,
    FailureClassificationConfig,
)
from nemo_gym.rollouts.rollout_outcomes import RolloutFailureRecord
from nemo_gym.rollouts.rollout_recovery import (
    RunManifest,
    _digest,
    _get_max_rollout_attempts,
    has_recorded_grade,
    has_saved_generation,
    has_unclassified_execution_failure,
    is_answered_agent_deadline,
    is_infrastructure_agent_timeout,
    is_terminal_failure,
    keeps_partial_response,
    load_attempt_extension,
    load_rerun_masked_approvals,
    load_resubmit_offset,
    manifest_path_for,
    masked_infrastructure_kind,
    masked_kept_answer_kind,
    matches_failure_class,
    observed_elapsed,
    saved_failure_classification,
)
from nemo_gym.utils.path_utils import failures_path_for, resubmit_offset_path_for


COUNTS_AS_ZERO_ROW_KEY = "_ng_failure_counts_as_zero"
RUN_ID_KEY = "_ng_run_id"
# Names the saved judge failure a judge-only re-judgment settles; the result keeps that failure's attempt identity.
REJUDGED_FAILURE_ID_KEY = "_ng_rejudged_failure_id"
# The pending_rejudge operation an outcome closes.
REJUDGE_OPERATION_ID_KEY = "_ng_rejudge_operation_id"
# The settled judge failure's own observations, kept on the selected result.
REJUDGED_OBSERVATIONS_KEY = "_ng_rejudged_failure_observations"
# A re-verified saved result is named by this prefix and the digest of the saved row (``rejudge_predecessor_id``).
RESULT_PREDECESSOR_PREFIX = "result-sha256:"


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
    # Top-level keys a read-only selection skips in large rows; ``read`` applies the same projection.
    skip_keys: frozenset[str] = frozenset()

    def read(self) -> dict:
        """Read this record with its effective legacy attempt identity."""
        with self.path.open("rb") as file:
            return self._read(file)

    def _read(self, file: BinaryIO) -> dict:
        file.seek(self.offset)
        if self.skip_keys:
            row, _ = _read_row(file, self.length, self.skip_keys)
            if row is None:
                raise ConfigError(f"Saved record at byte {self.offset} of {self.path} is incomplete.")
        else:
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
    # A scored result whose producer applied ``count_as_zero`` (a failure the policy settled as the model's zero): the
    # failure kind. It is a success to selection, so only the coverage report singles it out.
    counted_zero_kind: str | None = None
    interruption_cause_unknown: bool = False
    saved_answer: bool = False
    unclassified_execution_failure: bool = False
    # A result masked by an infrastructure failure (``masked_infrastructure_kind``): a failed attempt that is retried.
    masked_infrastructure: bool = False
    # The masked result an operator approved to run again (``masked_kept_answer_kind``): counted as the line above.
    rerun_approved: bool = False


# A read-only selection parses a row above this size without the values of its skipped keys, in chunks.
PROJECTED_ROW_BYTES = 64 << 20
_CHUNK_BYTES = 1 << 20
_STRUCTURE = re.compile(rb'["{}\[\],:]')
_NESTED = re.compile(rb'["{}\[\]]')
_IN_STRING = re.compile(rb'["\\]')
_NON_SPACE = re.compile(rb"[^ \t\r\n]")
# Inside a nested value only brackets matter: one match skips plain bytes and whole strings up to the next one.
_NESTED_RUN = re.compile(rb'[^"\\\[\]{}]*(?:"[^"\\]*(?:\\.[^"\\]*)*"[^"\\\[\]{}]*)*')


class _TopLevelProjector:
    """Copy one JSON object, replacing the values of some top-level keys with null, in bounded memory.

    Bytes are fed in chunks; a skipped value is scanned for its end but never kept. Only top-level keys are
    matched, so the same name nested anywhere else is copied.
    """

    def __init__(self, skip_keys: frozenset[str]) -> None:
        self._skip = {key.encode() for key in skip_keys}
        self._parts: list[bytes] = []
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._key: bytearray | None = None
        self._expect_key = False
        self._last_key: bytes | None = None
        self._await_value = False
        self._skipping = False

    def feed(self, chunk: bytes) -> None:
        pos, size = 0, len(chunk)
        copy_from = None if self._skipping else 0
        while pos < size:
            if self._in_string:
                if self._escape:
                    self._escape = False
                    if self._key is not None:
                        self._key += chunk[pos : pos + 1]
                    pos += 1
                    continue
                found = _IN_STRING.search(chunk, pos)
                end = found.start() if found else size
                if self._key is not None and len(self._key) <= 256:
                    self._key += chunk[pos:end]
                if found is None:
                    break
                pos = found.end()
                if found.group() == b"\\":
                    self._escape = True
                    continue
                self._in_string = False
                if self._key is not None:
                    self._last_key, self._key = bytes(self._key), None
                continue
            if self._await_value:
                found = _NON_SPACE.search(chunk, pos)
                if found is None:
                    break
                pos = found.start()
                self._await_value = False
                if self._last_key in self._skip:
                    self._parts.append(chunk[copy_from:pos] + b"null")
                    copy_from, self._skipping = None, True
                continue
            if self._depth >= 2:
                pos = _NESTED_RUN.match(chunk, pos).end()
                if pos >= size:
                    break
            found = (_STRUCTURE if self._depth <= 1 else _NESTED).search(chunk, pos)
            if found is None:
                break
            char, pos = found.group(), found.end()
            if char == b'"':
                self._in_string = True
                if self._depth == 1 and self._expect_key:
                    self._key, self._expect_key = bytearray(), False
            elif char in (b"{", b"["):
                self._depth += 1
                self._expect_key = self._depth == 1 and char == b"{"
            elif char in (b"}", b"]"):
                if self._depth == 1 and self._skipping:
                    self._skipping, copy_from = False, found.start()
                self._depth -= 1
            elif char == b"," and self._depth == 1:
                if self._skipping:
                    self._skipping, copy_from = False, found.start()
                self._expect_key = True
            elif char == b":" and self._depth == 1 and self._last_key is not None:
                self._await_value = True
        if copy_from is not None:
            self._parts.append(chunk[copy_from:])

    def result(self) -> bytes:
        return b"".join(self._parts)


def _read_row(file: BinaryIO, limit: int | None, skip_keys: frozenset[str]) -> tuple[dict | None, int]:
    """Read one JSONL row from the file position in chunks; a large row is parsed without its skipped values.

    Returns the row (None for a blank line, or for an incomplete final row) and the bytes consumed.
    """
    kept: list[bytes] = []
    projector = None
    consumed = 0
    ended = False
    while limit is None or consumed < limit:
        chunk = file.readline(_CHUNK_BYTES if limit is None else min(_CHUNK_BYTES, limit - consumed))
        if not chunk:
            break
        consumed += len(chunk)
        ended = chunk.endswith(b"\n")
        if projector is not None:
            projector.feed(chunk)
        else:
            kept.append(chunk)
            if consumed > PROJECTED_ROW_BYTES:
                projector = _TopLevelProjector(skip_keys)
                for part in kept:
                    projector.feed(part)
                kept = []
        if ended:
            break
    raw = projector.result() if projector is not None else b"".join(kept)
    if not raw.strip():
        return None, consumed
    if not ended:
        raise _IncompleteRow(consumed)
    value = orjson.loads(raw)
    for key in skip_keys if projector is not None else ():
        value.pop(key, None)
    return value, consumed


class _IncompleteRow(Exception):
    def __init__(self, consumed: int) -> None:
        self.consumed = consumed


def _indexed_records(
    path: Path, *, end_offset: int | None = None, skip_keys: frozenset[str] = frozenset()
) -> Iterator[tuple[RolloutRecord, dict]]:
    """Scan one record at a time, retaining its original byte coordinates.

    With ``skip_keys`` a row larger than ``PROJECTED_ROW_BYTES`` is read in chunks and parsed without those
    top-level values, so a read-only selection never holds a whole large row.
    """
    if not path.exists():
        return
    with path.open("rb") as file:
        stat = os.fstat(file.fileno())
        file_identity = (stat.st_dev, stat.st_ino)
        offset = 0
        number = 0
        while end_offset is None or offset < end_offset:
            limit = None if end_offset is None else end_offset - offset
            if skip_keys:
                try:
                    value, length = _read_row(file, limit, skip_keys)
                except _IncompleteRow:
                    warnings.warn(f"Ignoring incomplete final record in {path} at line {number + 1}.", stacklevel=2)
                    return
                except orjson.JSONDecodeError as error:
                    raise ConfigError(f"Malformed JSON in {path} at line {number + 1}: {error}") from error
                if not length:
                    break
                number += 1
                record = RolloutRecord(path, offset, length, number, file_identity=file_identity, skip_keys=skip_keys)
                offset += length
                if value is None:
                    continue
            else:
                raw = file.readline(-1 if limit is None else limit)
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


def rejudge_predecessor_id(row: dict) -> str | None:
    """The id a re-judgment names its predecessor by: a failure's ``_ng_failure_id``, else, for a saved result whose
    grading is re-verified from its saved answer, the digest of the whole saved row."""
    if row.get("_ng_failure_class") is not None:
        failure_id = row.get("_ng_failure_id")
        return failure_id if isinstance(failure_id, str) and failure_id else None
    return RESULT_PREDECESSOR_PREFIX + hashlib.sha256(orjson.dumps(row, option=orjson.OPT_SORT_KEYS)).hexdigest()


def rejudgment_admission_closed_by(
    pending, outcome: dict, identity: str, failures_path: Path, results_path: Path | None = None
) -> bool:
    """Whether outcome is the saved result of admission pending: operation, predecessor id and row hash, identity,
    attempt and admission time all match. Cold readers and the close use this one check.

    A failure predecessor is found in ``failures_path``; a saved-result predecessor in ``results_path``."""
    if (
        outcome.get(REJUDGE_OPERATION_ID_KEY) != pending.operation_id
        or outcome.get(REJUDGED_FAILURE_ID_KEY) != pending.predecessor_failure_id
        or outcome.get("_ng_rejudge_admitted_at") != pending.admitted_at
    ):
        return False
    result_predecessor = pending.predecessor_failure_id.startswith(RESULT_PREDECESSOR_PREFIX)
    if result_predecessor and results_path is None:
        return False
    for _, row in _indexed_records(results_path if result_predecessor else failures_path):
        if rejudge_predecessor_id(row) == pending.predecessor_failure_id:
            return (
                logical_rollout_id(row) == identity
                and row.get(ATTEMPT_INDEX_KEY_NAME, 0) == outcome.get(ATTEMPT_INDEX_KEY_NAME, 0)
                and hashlib.sha256(orjson.dumps(row, option=orjson.OPT_SORT_KEYS)).hexdigest()
                == pending.predecessor_sha256
            )
    return False


def is_rejudgment_of(failure: dict, result: dict) -> bool:
    """Whether ``result`` re-scores the saved answer of ``failure`` without a new model attempt.

    ``failure`` is a judge failure, or a saved result whose grading is re-verified from its saved answer (a result
    masked by a judge or verify-transport failure, or one with no recorded grading). Which saved results are
    re-verified is the caller's selection; this checks only that ``result`` binds the same saved answer and attempt.
    """
    predecessor_id = rejudge_predecessor_id(failure)
    return (
        failure.get("_ng_failure_class") in ("judge_failed", None)
        and _holds_saved_answer(failure)
        and predecessor_id is not None
        and result.get("_ng_failure_class") is None
        and type(result.get("reward")) in (int, float)
        and math.isfinite(result["reward"])
        and result.get("evaluation_completed") is not False
        and not result.get("mask_sample")
        and result.get(RUN_ID_KEY) == failure.get(RUN_ID_KEY)
        and result.get(ATTEMPT_INDEX_KEY_NAME, 0) == failure.get(ATTEMPT_INDEX_KEY_NAME, 0)
        and result.get(REJUDGED_FAILURE_ID_KEY) == predecessor_id
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
    def __init__(
        self, manifest: RunManifest, expected: list[dict], *, classification: FailureClassificationConfig | None = None
    ):
        self.classification = classification
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
        # Attempts an explicit recovery added to every rollout's saved cap (`--extend-attempts`).
        self.attempt_extension = 0
        # The masked attempts an operator approved to run again (`--rerun-masked-without-artifact`), by (task, rollout,
        # attempt): each only for the attempt it names.
        self.rerun_masked: frozenset[tuple[int, int, int]] = frozenset()

    @property
    def attempt_cap(self) -> int:
        """Every rollout's attempt cap: the run's saved cap plus the extension an explicit recovery recorded."""
        return (self.manifest.max_rollout_attempts or _get_max_rollout_attempts()) + self.attempt_extension

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
        infrastructure = is_infrastructure_agent_timeout(row, classification=self.classification)
        failure_class = row.get("_ng_failure_class")
        if failure_class is None and infrastructure:
            # A graded result of an agent timeout whose window held a server-side fault (Harbor's graded zero) is not
            # the model's result: it is selected as a failure, spends an attempt and runs again under the cap.
            failure_class = AGENT_TIMEOUT_INFRASTRUCTURE
        if (
            count_failure
            and key not in self.payloads
            and failure_class is not None
            and not interruption
            and not row.get("_ng_failure_terminal")
        ):
            self.attempt_counts[key[0]] += 1
        if (
            count_failure
            and key not in self.payloads
            and failure_class is not None
            and not interruption
            and row.get("_ng_failure_terminal")
            and not is_terminal_failure(row, retry_terminal_timeouts=True)
        ):
            self.overridden_terminal_counts[key[0]] += 1
        # A masked result that kept its answer runs again only when its attempt was approved: then it is retried as
        # any infrastructure-masked result. The approval never widens the rule; the row must still qualify.
        rerun_approved = (
            row.get(TASK_INDEX_KEY_NAME),
            row.get(ROLLOUT_INDEX_KEY_NAME),
            key[1],
        ) in self.rerun_masked and masked_kept_answer_kind(row, classification=self.classification) is not None
        masked_infrastructure = (
            masked_infrastructure_kind(row, classification=self.classification) is not None or rerun_approved
        )
        if masked_infrastructure and key not in self.payloads:
            # The results file keeps no resubmit offset, so a masked attempt counts whenever it was saved: a resubmit
            # still gives the process it starts a fresh budget, as for every failure.
            self.attempt_counts[key[0]] += 1
        if row.get("_ng_omitted"):
            self.omitted.add(key)
        self.payloads[key] = _Outcome(
            record=replace(record, expected_identity=(row.get(RUN_ID_KEY), key[0], key[1])),
            failure_class=failure_class,
            legacy_failure_class=row.get("_ng_failure_legacy_class"),
            terminal=bool(row.get("_ng_failure_terminal")),
            masked=bool(row.get("mask_sample")),
            has_reward=type(row.get("reward")) in (int, float),
            elapsed_seconds=observed_elapsed(row),
            # The producer settled this rollout as a scored zero, or the agent's deadline ended it with its answer
            # kept (the model's result, as recovery classifies it): final, so no policy reopens it.
            final_zero=(row.get("_ng_failure_counts_as_zero") is True and bool(row.get("_ng_failure_terminal")))
            or is_answered_agent_deadline(row, classification=self.classification),
            counted_zero_kind=(
                str(row.get("failure_kind") or "unknown")
                if failure_class is None
                and (row.get("failure_outcome_action") == "count_as_zero" or row.get(COUNTS_AS_ZERO_ROW_KEY) is True)
                else None
            ),
            interruption_cause_unknown=interruption and not _has_eligible_interruption_cause(row),
            unclassified_execution_failure=has_unclassified_execution_failure(row),
            # A recorded grade or a partial native response is retained like a kept answer: a fresh run would
            # replace it. Not when a server-side fault cut the agent's window short: that answer is not its result.
            saved_answer=failure_class is not None
            and not infrastructure
            and (_holds_saved_answer(row) or has_recorded_grade(row) or keeps_partial_response(row)),
            masked_infrastructure=masked_infrastructure,
            rerun_approved=rerun_approved,
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
        skip_keys: frozenset[str] = frozenset(),
    ) -> "RolloutRecords":
        """Index outcomes, optionally restricted to prefixes captured before the manifest read.

        ``skip_keys`` names top-level row values that no selection rule reads; large rows are parsed without them.
        """
        expected = list(read_records(materialized_path_for(output)))
        if _digest(expected) != manifest.materialized_digest:
            raise ConfigError("Saved materialized inputs do not match the run manifest.")
        state = cls(manifest, expected, classification=saved_failure_classification(output))
        unexpected = manifest.next_attempt.keys() - state.expected.keys()
        if unexpected:
            raise ConfigError(f"Run manifest reserves attempts for unknown rollouts: {sorted(unexpected)!r}.")
        # Legacy files lacked an attempt id on some rows. Import once in their
        # recorded order, preserving the old failure-count-based numbering.
        legacy_counts: Counter = Counter()
        state.attempt_extension = load_attempt_extension(output)
        state.rerun_masked = load_rerun_masked_approvals(output)
        resubmit_offset = load_resubmit_offset(output)
        resubmitted = resubmit_offset_path_for(output).exists()
        for path in (failures_path_for(output), output):
            for record, payload in _indexed_records(
                path, end_offset=outcome_limits[path] if outcome_limits is not None else None, skip_keys=skip_keys
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
                pending, latest.record.read(), identity, failures_path_for(output), output
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
        if payload and payload.masked_infrastructure and self.disposition(identity) == "success":
            # Infrastructure masked this result, so it is retried and a later attempt supersedes it.
            return True
        return self.disposition(identity) not in {"success", "omitted"} and not terminal

    def rerun_approved_identities(self) -> set[str]:
        """The logical rollouts whose selected result is a masked one an operator approved to run again."""
        return {
            identity
            for identity, index in self.latest.items()
            if (payload := self.payloads.get((identity, index))) is not None
            and payload.rerun_approved
            and self.disposition(identity) == "success"
        }

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

    def allocation_end_charges(self) -> list[dict]:
        """The failure rows that charge a rollout for each later end of an allocation it was in flight in.

        A reserved attempt with no saved outcome was in flight when the allocation that made it ended: the writer that
        loads the Store holds run_lock, so no other allocation of the run is alive. The first such end of an identity
        re-runs for free, and every later one spends an attempt of its cap: the earliest reservation without an
        outcome stays as it is and each later one gets a non-terminal ``rollout_allocation_ended`` failure. Only an
        identity a resume would run again is charged; a finished, settled or blocked one is never touched. The count
        is derived from the manifest's reservations each time, so a saved charge is never charged again.
        """
        if self.manifest.legacy_import:
            return []
        charges = []
        for identity, original in self.expected.items():
            if not self._retryable(identity):
                continue
            ended = [
                index
                for index in range(self.manifest.next_attempt.get(identity, 0))
                if (identity, index) not in self.payloads
            ]
            for index in ended[1:]:
                charges.append(
                    dict(original)
                    | {
                        ATTEMPT_INDEX_KEY_NAME: index,
                        RUN_ID_KEY: self.manifest.run_id,
                        "_ng_failure_class": ROLLOUT_ALLOCATION_ENDED,
                        "_ng_failure_terminal": False,
                        "failure_kind": ROLLOUT_ALLOCATION_ENDED,
                        "_ng_failure_message": (
                            f"attempt {index} was in flight when its allocation ended; this rollout's first such end "
                            "re-ran for free and each later one spends an attempt"
                        ),
                    }
                )
        return charges

    def stopped_count(self, max_attempts: int) -> int:
        """Failed rollouts resume does not run again although they are neither final nor at their cap: a kept answer
        or a judge failure (left to verification recovery) and an unclassified execution (left to recovery)."""
        return sum(
            self.disposition(identity) == "failure"
            and not self._final(identity)
            and self.failure_count(identity) < max_attempts
            and not self._retryable(identity)
            for identity in self.expected
        )

    def _final(self, identity: str) -> bool:
        payload = self.payloads.get((identity, self.latest.get(identity)))
        return payload is not None and (payload.terminal or payload.final_zero)

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
        max_attempts = self.attempt_cap
        not_started = expected - len(self.latest)
        missing = counts["unknown"] - not_started
        # A producer-masked result completed execution, so recovery reuses it unless infrastructure masked it
        # (``masked_infrastructure``). Report its measurement status separately, after selecting attempts.
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
        # Scored successes the policy counted as the model's zero are already in ``successful``: they are reported
        # apart from ``final_zeros`` (failed rollouts only), so no completion formula counts them twice.
        scored_zeros = Counter(
            self.payloads[(identity, self.latest[identity])].counted_zero_kind
            for identity in self.expected
            if self.disposition(identity) == "success"
            and self.payloads[(identity, self.latest[identity])].counted_zero_kind
        )
        failed_final = sum(
            not self._retryable(identity) or self.failure_count(identity) >= max_attempts for identity, _ in failures
        )
        # A failure record whose producer set mask_sample (a judge or verifier outage, for example) is excluded from
        # the score rather than counted as a zero. It is a failure, so it is never in ``masked``: report it here.
        failed_masked = sum(self.payloads[key].masked for key in failures)
        # Masked by infrastructure and still within its attempt cap: a resume runs it again, so it is not complete.
        masked_retryable = sum(
            self.payloads[(identity, self.latest[identity])].masked_infrastructure
            and self._retryable(identity)
            and self.failure_count(identity) < max_attempts
            for identity in self.expected
            if self.disposition(identity) == "success"
        )
        settled = counts["success"] - masked_retryable + final_zeros
        # Rollouts with no final result: a resume still runs them, or nothing has run for them yet. Final failures
        # (terminal, at their attempt cap, or ``stopped_not_retried``, which resume does not run again) are not open,
        # even when they are not settled.
        open_rollouts = masked_retryable + (len(failures) - failed_final) + not_started + missing
        return {
            "schema_version": 2,
            "selection_policy": self.manifest.selection_policy,
            "run_id": self.manifest.run_id,
            "expected": expected,
            "successful": counts["success"],
            "measured": counts["success"] - masked - unscored,
            "masked": masked,
            "masked_retryable": masked_retryable,
            "unscored": unscored,
            "failed": counts["failure"],
            "failed_final": failed_final,
            "failed_masked": failed_masked,
            "final_zeros": final_zeros,
            "counted_zeros_in_successful": sum(scored_zeros.values()),
            "counted_zero_kinds": dict(sorted(scored_zeros.items())),
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
            # Of failed_final: failed rollouts neither terminal nor at their cap that resume still does not run again,
            # left to (verification) recovery.
            "stopped_not_retried": self.stopped_count(max_attempts),
            "retryable": sum(
                self._retryable(identity) and self.failure_count(identity) < max_attempts for identity in self.expected
            ),
            "open": open_rollouts,
            "max_rollout_attempts": max_attempts,
            "attempt_extension": self.attempt_extension,
            "retry_terminal_timeouts": self.retry_terminal_timeouts,
            "completion_fraction": settled / expected if expected else 1.0,
            "complete": settled == expected,
            "reconciled": missing == 0,
            "identity_verified": not (self.manifest.legacy_import or self.manifest.identity_overridden),
        }
