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
"""Own the evaluation runner's artifacts and their recovery protocol.

One writer owns each output path. Request dispatch, capture finalization and
scoring policy belong to the caller; this store owns file lifetime and ordering.
The journal reconstructs state for both collection and read-only aggregation.
"""

import os
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import orjson

from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_journal import (
    RUN_ID_KEY,
    RolloutJournal,
    coverage_path_for,
    journal_path_for,
    logical_rollout_id,
    materialized_path_for,
    prepare_append,
    read_records,
)
from nemo_gym.rollout_recovery import RunManifest, atomic_write_json, manifest_path_for, validate_resume


class RolloutStore:
    def __init__(self, output: Path, state: RolloutJournal, *, seed_legacy: bool = False, read_only: bool = False):
        self.output = output
        self.manifest = state.manifest
        self._state = state
        self._seed_legacy = seed_legacy
        self._read_only = read_only
        self._files = None
        self._results_file = None
        self._failures_file = None

    @classmethod
    def start_or_resume(
        cls,
        output: Path,
        prepare_inputs: Callable[[], tuple[list[dict], RunManifest]],
        *,
        resume: bool,
        allow_unsafe: bool = False,
    ) -> "RolloutStore":
        """Validate saved work before mutation; materialize fresh work once.

        The callback keeps dataset/configuration preparation in the collector.
        Legacy imports need not have their original dataset available, so they
        deliberately do not invoke it when no saved manifest exists.
        """
        manifest_path = manifest_path_for(output)
        materialized = materialized_path_for(output)
        journal = journal_path_for(output)
        artifacts = (output, failures_path_for(output), materialized, manifest_path, journal)
        output.parent.mkdir(parents=True, exist_ok=True)
        if resume and any(path.exists() for path in artifacts):
            if not materialized.exists() or not output.exists():
                raise ConfigError("Cannot resume: saved materialized inputs or rollout output are missing.")
            current = None
            if manifest_path.exists():
                try:
                    _, current = prepare_inputs()
                except (ConfigError, OSError):
                    if not allow_unsafe:
                        raise
            manifest = validate_resume(manifest_path, current, materialized, allow_unsafe=allow_unsafe)
            seed_legacy = False
            if manifest is None:
                manifest = RunManifest.import_legacy(list(read_records(materialized)))
                seed_legacy = True
            elif not journal.exists():
                if not allow_unsafe:
                    raise ConfigError(f"Cannot resume without attempt history: {journal}.")
                manifest = manifest.model_copy(update={"legacy_import": True})
                seed_legacy = True
            state = RolloutJournal.load(output, manifest, import_legacy=manifest.legacy_import)
            if seed_legacy or manifest.identity_overridden:
                manifest.write(manifest_path)
            return cls(output, state, seed_legacy=seed_legacy)

        rows, manifest = prepare_inputs()
        state = RolloutJournal(manifest, rows)
        with materialized.open("wb") as file:
            for row in rows:
                file.write(orjson.dumps(row) + b"\n")
        # Invalidate prior outputs before publishing the fresh run's identity.
        for path in (output, failures_path_for(output), journal, coverage_path_for(output)):
            path.unlink(missing_ok=True)
        manifest.write(manifest_path)
        return cls(output, state)

    @classmethod
    def read(cls, output: Path) -> "RolloutStore | None":
        """Read the same selected outcomes offline, without modifying artifacts.

        A legacy file without an input inventory has unknown completion coverage;
        return None so its caller can choose its documented compatibility path.
        """
        path = manifest_path_for(output)
        if path.exists():
            manifest = RunManifest.model_validate_json(path.read_bytes())
        elif materialized_path_for(output).exists():
            manifest = RunManifest.import_legacy(list(read_records(materialized_path_for(output))))
        else:
            return None
        state = RolloutJournal.load(output, manifest, import_legacy=manifest.legacy_import)
        return cls(output, state, read_only=True)

    def __enter__(self) -> "RolloutStore":
        if self._read_only or self._files is not None:
            raise RuntimeError("Cannot open a read-only or already open rollout store for writing.")
        files = ExitStack()
        try:
            journal = journal_path_for(self.output)
            failures = failures_path_for(self.output)
            for path in (self.output, failures, journal):
                prepare_append(path)
            # Registered first, so the snapshot follows file closure on every exit.
            files.callback(self.write_coverage)
            self._state.file = files.enter_context(journal.open("ab"))
            self._results_file = files.enter_context(self.output.open("ab"))
            self._failures_file = files.enter_context(failures.open("ab"))
            if self._seed_legacy:
                self._state.seed_legacy_history()
                self._seed_legacy = False
            self.write_coverage()
        except BaseException:
            try:
                files.close()
            finally:
                self._state.file = None
                self._results_file = self._failures_file = None
            raise
        self._files = files
        return self

    def __exit__(self, *exc):
        try:
            return self._files.__exit__(*exc)
        finally:
            self._files = None
            self._state.file = None
            self._results_file = self._failures_file = None

    def _require_open(self) -> None:
        if self._files is None:
            raise RuntimeError("Rollout persistence requires an open store context.")

    def record_dispatch(self, row: dict) -> None:
        self._require_open()
        self._state.dispatch(row)

    def record_outcome(self, result: dict, *, sync: bool = False) -> None:
        """Commit a payload before its outcome event; reject conflicts before writing.

        A crash between those writes is recoverable from the dispatch and payload.
        Callers retiring external capture evidence request fsync before retirement.
        """
        self._require_open()
        self._state.check_outcome(result)
        file = self._failures_file if result.get("_ng_failure_class") is not None else self._results_file
        file.write(orjson.dumps(result) + b"\n")
        file.flush()
        if sync:
            os.fsync(file.fileno())
        self._state.outcome(result)

    def record_omission(self, row: dict, reason: str) -> None:
        self._require_open()
        self._state.omit(row, reason)

    def pending(self, max_attempts: int) -> list[dict]:
        # The caller supplies its attempt budget; the stored selection policy is
        # latest_dispatched, including newer attempts with unknown outcomes.
        return [dict(row, **{RUN_ID_KEY: self.manifest.run_id}) for row in self._state.pending(max_attempts)]

    def selected(self, disposition: str) -> list[dict]:
        return self._state.selected(disposition)

    def inputs_for(self, results: list[dict]) -> list[dict]:
        return [self._state.expected[logical_rollout_id(result)] for result in results]

    def coverage(self) -> dict:
        return self._state.coverage()

    def write_coverage(self, **reporting: int) -> None:
        atomic_write_json(coverage_path_for(self.output), self.coverage() | reporting)
