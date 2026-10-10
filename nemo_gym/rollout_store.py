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
"""Persist evaluation outcomes and reserve execution identities for its controller.

The controller holds run_lock across preparation, optional checkpoint restoration,
collection and reporting. Results and sidecars contain outcomes; the manifest owns
execution numbers. There is no dispatch journal or second outcome commit.
"""

import logging
import os
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import orjson
from pydantic import ValidationError

from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME
from nemo_gym.path_utils import aggregate_metrics_path_for, failures_path_for
from nemo_gym.rollout_records import (
    RUN_ID_KEY,
    RolloutRecord,
    RolloutRecords,
    _indexed_records,
    coverage_path_for,
    journal_path_for,
    logical_rollout_id,
    materialized_path_for,
    prepare_append,
    read_records,
    resolve_rollout_owner,
    resolve_rollout_path,
)
from nemo_gym.rollout_recovery import RunManifest, atomic_write_json, manifest_path_for, validate_resume


logger = logging.getLogger(__name__)


def raw_outcomes_are_selected(path: Path) -> bool:
    """Whether an existing row-oriented reader can read this run without filtering.

    A manifest alone does not require a different reader. Superseded rows,
    including answers superseded by interrupted attempts, do. Validate the whole
    run even when the caller reads only the success file or only its sidecar.
    """
    output = resolve_rollout_owner(path)
    if journal_path_for(output).exists():
        return False
    if not manifest_path_for(output).exists():
        return True
    store = RolloutStore.read(output)
    selected = {
        (record.path, record.offset)
        for disposition in ("success", "failure", "omitted")
        for record in store.selected_records(disposition).values()
    }
    raw = {
        (record.path, record.offset)
        for artifact in (output, failures_path_for(output))
        for record, _ in _indexed_records(artifact)
    }
    return raw == selected


class RolloutStore:
    """Saved outcomes shared by the evaluation controller and offline readers."""

    def __init__(self, output: Path, state: RolloutRecords, *, read_only: bool = False):
        self.output = output
        self.manifest = state.manifest
        self._state = state
        self._read_only = read_only
        self._files = None
        self._results_file = self._failures_file = None
        self._allocated: set[tuple[str, int]] = set()
        self.outcomes_recorded: int = 0

    @staticmethod
    def _unverified_manifest(output: Path) -> RunManifest:
        manifest = RunManifest.import_legacy(list(read_records(materialized_path_for(output))))
        run_ids = {
            row[RUN_ID_KEY]
            for path in (output, failures_path_for(output))
            for row in read_records(path)
            if row.get(RUN_ID_KEY) is not None
        }
        if len(run_ids) > 1:
            raise ConfigError("Saved artifacts belong to different runs.")
        if run_ids:
            # Missing execution reservations cannot safely be inferred from outcomes:
            # a lost manifest may hide newer attempts with no outcome.
            raise ConfigError(
                "Run-tagged outcomes have lost their manifest. Restore it from backup or use a new output path."
            )
        return manifest

    @classmethod
    def start_or_resume(
        cls,
        output: Path,
        prepare_inputs: Callable[[], tuple[list[dict], RunManifest]],
        *,
        resume: bool,
        allow_unsafe: bool = False,
        migrate_outcomes: Callable[[Path], int] | None = None,
    ) -> "RolloutStore":
        """Prepare or validate a run while its controller holds the run lock."""
        output = resolve_rollout_path(output)
        manifest_path = manifest_path_for(output)
        materialized = materialized_path_for(output)
        artifacts = (output, failures_path_for(output), materialized, manifest_path, journal_path_for(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        if resume and journal_path_for(output).exists():
            raise ConfigError(
                "This run uses the superseded draft dispatch-journal format. Use its original Gym revision or start at a new output path."
            )
        if (
            resume
            and any(path.exists() for path in artifacts)
            and not (materialized.exists() and output.exists())
            and not manifest_path.exists()
        ):
            # Missing companions do not make tagged outcomes an uninitialized
            # cache. Never erase evidence of an established run during resume.
            if any(
                row.get(RUN_ID_KEY) is not None
                for path in (output, failures_path_for(output))
                for row in read_records(path)
            ):
                raise ConfigError(
                    "Run-tagged outcomes have lost their manifest. Restore it from backup or use a new output path."
                )
            if any(path.exists() and path.stat().st_size for path in (output, failures_path_for(output))):
                raise ConfigError(
                    "Cannot resume: the legacy cache is incomplete and contains saved outcomes. "
                    "Restore the missing files or choose a new output path. Saved artifacts were not changed."
                )
            print("Skipping resume_from_cache because the legacy cache is incomplete; starting fresh.")
            resume = False
        if resume and any(path.exists() for path in artifacts):
            if (
                not materialized.exists()
                or not output.exists()
                or (manifest_path.exists() and not failures_path_for(output).exists())
            ):
                raise ConfigError(
                    "Cannot resume: saved materialized inputs, rollout output, or failure sidecar are missing."
                )
            current = None
            if manifest_path.exists():
                try:
                    _, current = prepare_inputs()
                except (ConfigError, OSError):
                    if not allow_unsafe:
                        raise
            manifest = validate_resume(manifest_path, current, materialized, allow_unsafe=allow_unsafe)
            if manifest is None:
                manifest = cls._unverified_manifest(output)
            state = RolloutRecords.load(output, manifest, import_legacy=manifest.legacy_import)
            if manifest.legacy_import:
                next_attempt = dict(manifest.next_attempt)
                for identity, attempt in state.payloads:
                    next_attempt[identity] = max(next_attempt.get(identity, 0), attempt + 1)
                manifest = manifest.model_copy(update={"next_attempt": next_attempt})
                state.manifest = manifest
            if migrate_outcomes is not None:
                for path in (output, failures_path_for(output)):
                    prepare_append(path)
                if migrate_outcomes(output):
                    state = RolloutRecords.load(output, manifest, import_legacy=manifest.legacy_import)
            if manifest.identity_overridden or manifest.legacy_import or not manifest_path.exists():
                manifest.write(manifest_path)
            return cls(output, state)

        rows, manifest = prepare_inputs()
        state = RolloutRecords(manifest, rows)
        with materialized.open("wb") as file:
            for row in rows:
                file.write(orjson.dumps(row) + b"\n")
        for path in (
            output,
            failures_path_for(output),
            journal_path_for(output),
            coverage_path_for(output),
            aggregate_metrics_path_for(output),
        ):
            path.unlink(missing_ok=True)
        output.touch()
        failures_path_for(output).touch()
        manifest.write(manifest_path)
        return cls(output, state)

    @classmethod
    def read(
        cls, output: Path, *, import_legacy: bool = True, retry_terminal_timeouts: bool = False
    ) -> "RolloutStore | None":
        """Read selected outcomes without modifying files or acquiring a writer lock."""
        output = resolve_rollout_path(output, read_only=True)
        path = manifest_path_for(output)
        if journal_path_for(output).exists():
            raise ConfigError(
                "This run uses the superseded draft dispatch-journal format; read it with its original Gym revision."
            )
        if not import_legacy and not path.exists():
            return None
        if path.exists():
            if not output.exists() or not failures_path_for(output).exists():
                raise ConfigError(f"Cannot read run {output}: rollout output or failure sidecar is missing.")
            try:
                manifest = RunManifest.model_validate_json(path.read_bytes())
            except (ValidationError, OSError) as error:
                raise ConfigError(f"Cannot read recovery manifest {path}: {error}") from error
        elif materialized_path_for(output).exists():
            manifest = cls._unverified_manifest(output)
        else:
            return None
        state = RolloutRecords.load(output, manifest, import_legacy=manifest.legacy_import)
        state.retry_terminal_timeouts = retry_terminal_timeouts
        return cls(output, state, read_only=True)

    def __enter__(self) -> "RolloutStore":
        if self._read_only or self._files is not None:
            raise RuntimeError("Cannot open a read-only or already open rollout store for writing.")
        files = ExitStack()
        try:
            for path in (self.output, failures_path_for(self.output)):
                prepare_append(path)
            self._results_file = files.enter_context(self.output.open("ab"))
            self._failures_file = files.enter_context(failures_path_for(self.output).open("ab"))
            self.write_coverage()
        except BaseException:
            files.close()
            self._results_file = self._failures_file = None
            raise
        self._files = files
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._files.__exit__(exc_type, exc_value, traceback)
        finally:
            self._files = None
            self._results_file = self._failures_file = None
            try:
                self.write_coverage()
            except Exception:
                if exc_type is None:
                    raise
                logger.exception("Could not write coverage while handling collection failure")

    def _require_open(self) -> None:
        if self._files is None:
            raise RuntimeError("Rollout persistence requires an open store context.")

    def allocate_attempt(self, row: dict) -> int:
        """Durably reserve the next execution number before dispatch or restore.

        The controller passes the returned number to checkpoint restore; restore
        must not increment it. A failed restore reserves a new number before a
        fresh execution. Reservations do not consume the failure retry budget.
        """
        self.allocate_attempts([row])
        return row[ATTEMPT_INDEX_KEY_NAME]

    def allocate_attempts(self, rows: list[dict]) -> None:
        """Reserve one admitted batch atomically before any of its requests start."""
        self._require_open()
        if not rows:
            return
        next_attempt = dict(self.manifest.next_attempt)
        reservations = []
        identities = set()
        for row in rows:
            identity, _ = self._state._key(row)
            if identity in identities:
                raise ConfigError(f"Duplicate rollout in reservation batch: {identity}")
            identities.add(identity)
            attempt = next_attempt.get(identity, self._state.latest.get(identity, -1) + 1)
            reservations.append((row, identity, attempt))
            next_attempt[identity] = attempt + 1
        manifest = self.manifest.model_copy(update={"next_attempt": next_attempt})
        manifest.write(manifest_path_for(self.output))
        # Publish in memory only after all reservations are durable.
        self.manifest = self._state.manifest = manifest
        for row, identity, attempt in reservations:
            self._state.latest[identity] = attempt
            row[ATTEMPT_INDEX_KEY_NAME] = attempt
            row[RUN_ID_KEY] = manifest.run_id
            self._allocated.add((identity, attempt))

    def record_dispatch(self, row: dict) -> None:
        """Reserve once for this controller's request; no dispatch event is stored."""
        self.record_dispatches([row])

    def record_dispatches(self, rows: list[dict]) -> None:
        """Reserve requests admitted together with one durable manifest write."""
        self._require_open()
        self.allocate_attempts([row for row in rows if self._state._key(row) not in self._allocated])

    def record_outcome(self, result: dict, *, sync: bool = False) -> None:
        """Append the outcome once; a complete JSONL row is the recovery record."""
        self._require_open()
        self._state.check_outcome(result)
        file = self._failures_file if result.get("_ng_failure_class") is not None else self._results_file
        raw = orjson.dumps(result) + b"\n"
        stat = os.fstat(file.fileno())
        record = RolloutRecord(Path(file.name), file.tell(), len(raw), file_identity=(stat.st_dev, stat.st_ino))
        file.write(raw)
        file.flush()
        if sync:
            os.fsync(file.fileno())
        self._state.outcome(result, record=record)
        self.outcomes_recorded += 1

    def record_omission(self, row: dict, reason: str) -> None:
        """Save intentional omissions separately from scored results and failures."""
        self.record_dispatch(row)
        self.record_outcome(
            row
            | {
                "_ng_failure_class": "skipped",
                "_ng_failure_terminal": True,
                "_ng_omitted": True,
                "_ng_failure_message": reason[:2000],
            }
        )

    def pending(
        self, max_attempts: int, *, retry_terminal_timeouts: bool = False, dispatch_longest_first: bool = False
    ) -> list[dict]:
        # The caller supplies its attempt budget; the stored selection policy is
        # latest_allocated, including newer attempts with unknown outcomes.
        self._state.retry_terminal_timeouts = retry_terminal_timeouts
        exhausted = self._state.exhausted_count(max_attempts)
        if exhausted:
            print(
                f"Retry budget exhausted for {exhausted} rollout(s) at the cap of {max_attempts} counted failures. "
                f"They will not be dispatched with this cap. Failures: {failures_path_for(self.output)}."
            )
        rows = [dict(row, **{RUN_ID_KEY: self.manifest.run_id}) for row in self._state.pending(max_attempts)]
        elapsed = {}
        # Main's cached-deliverable and longest-first controls still apply when
        # the manifest supplies the attempt identities.
        # Unknown attempts carry no new decision about a cached deliverable.
        # Stop at the newest recorded outcome (including an omission), rather
        # than carrying a reuse instruction past a newer failure without one.
        latest_recorded = {}
        for identity, attempt in self._state.payloads.keys() | self._state.omitted:
            latest_recorded[identity] = max(attempt, latest_recorded.get(identity, -1))
        for row in rows:
            identity = logical_rollout_id(row)
            previous = self._state.payloads.get((identity, latest_recorded.get(identity)))
            if (
                previous is not None
                and (identity, latest_recorded.get(identity)) not in self._state.omitted
                and previous.reuse_cached_deliverable
            ):
                row["reuse_cached_deliverable"] = True
        if dispatch_longest_first:
            for (identity, _), outcome in self._state.payloads.items():
                if outcome.failure_class is not None:
                    duration = outcome.elapsed_seconds
                    if duration is not None:
                        elapsed[identity] = max(elapsed.get(identity, 0), duration)
            rows.sort(key=lambda row: -elapsed.get(logical_rollout_id(row), 0))
        return rows

    def attempt_count(self, row: dict) -> int:
        return self._state.attempt_counts[logical_rollout_id(row)]

    def disposition(self, row: dict) -> str:
        """Return the newest allocated attempt's disposition for this rollout."""
        return self._state.disposition(logical_rollout_id(row))

    def selected(self, disposition: str) -> list[dict]:
        """Load selected payloads; use selected_records for metadata-only access."""
        return self._state.selected(disposition)

    def selected_records(self, disposition: str) -> dict[tuple[str, int], RolloutRecord]:
        """Map selected rollout/attempt identities to their original file slices."""
        return self._state.selected_records(disposition)

    def failures(self) -> list[dict]:
        """Latest failure payloads, including terminal skips classified as omitted."""
        return [row for row in self.selected("failure") + self.selected("omitted") if not row.get("_ng_omitted")]

    def inputs_for(self, results: list[dict]) -> list[dict]:
        return [self._state.expected[logical_rollout_id(result)] for result in results]

    def coverage(self) -> dict:
        return self._state.coverage()

    def write_coverage(self, **reporting: int) -> None:
        atomic_write_json(
            coverage_path_for(self.output), self.coverage() | reporting | {"outcomes_recorded": self.outcomes_recorded}
        )
