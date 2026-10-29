# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional

import orjson
from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter, ValidationError

from nemo_gym.config.config_types import DEPRECATED_UPLOAD_ROLLOUTS_KEY, RunExportConfig, UploadRolloutsConfigMixin
from nemo_gym.config.secret_utils import recursively_hide_secrets
from nemo_gym.exporters.metrics import flatten_aggregate_metrics
from nemo_gym.path_utils import exporter_run_id_path_for
from nemo_gym.run_artifacts import RunArtifacts


logger = logging.getLogger(__name__)
# Pydantic's boolean parsing, which the run configs that read these flags use.
_BOOL = TypeAdapter(bool)


class ExportState(str, Enum):
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ExportResult:
    """What one exporter did with one finished run."""

    state: ExportState
    exit_code: int = 0
    error: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def succeeded(cls, **detail: Any) -> "ExportResult":
        return cls(ExportState.SUCCEEDED, detail=detail)

    @classmethod
    def skipped(cls, **detail: Any) -> "ExportResult":
        return cls(ExportState.SKIPPED, detail=detail)

    @classmethod
    def failed(cls, error: str, *, exit_code: int = 1, **detail: Any) -> "ExportResult":
        return cls(ExportState.FAILED, exit_code=exit_code, error=error, detail=detail)


class Exporter(ABC):
    """Sends a finished run to a destination, or converts it to another format.

    An exporter receives the run's `RunArtifacts` and decides which of those files it reads. Its settings
    come from `config_model`, validated from the export command's own config before `export` is called.
    Importing a backend's SDK belongs inside `export`, so listing exporters never loads one.
    """

    # Identifier used on the command line and in logs, e.g. "atif".
    name: ClassVar[str]
    config_model: ClassVar[type[RunExportConfig]]

    def __init__(self, config: RunExportConfig) -> None:
        self.config = config

    @classmethod
    def from_global_config(cls, global_config_dict: DictConfig) -> "Exporter":
        """Build the exporter from the export command's resolved config, validating it first."""
        return cls(cls.config_model.model_validate(global_config_dict))

    @abstractmethod
    def export(self, artifacts: RunArtifacts) -> ExportResult:
        """Export the run. Report an expected failure in the result; raise only for a bug."""


def _live_run_id(artifacts: RunArtifacts, name: str) -> Optional[str]:
    """The backend run a live exporter recorded for this run, or None when there is none or it cannot be read."""
    try:
        return artifacts.exporter_run_id_fpath(name).read_text().strip() or None
    except OSError:
        return None


def _write_marker_atomically(marker: Path, run_id: str) -> None:
    """Publish the already-exported marker only once it is fully written, so a failed write leaves none."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(run_id)
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)


def _upload_intent(config: DictConfig) -> UploadRolloutsConfigMixin:
    """The upload settings in `config`, resolving only those values.

    A saved run config keeps its `${...}` interpolations unresolved, and resolving all of it would need the
    environment the run had. The canonical key wins over the deprecated one, as in the settings model, so a
    deprecated value that loses is never evaluated.
    """
    values = OmegaConf.to_container(config, resolve=False)
    for key in ("upload_rollouts", DEPRECATED_UPLOAD_ROLLOUTS_KEY):
        if key in values:
            values[key] = config[key]
            break
    return UploadRolloutsConfigMixin.model_validate(values)


def _aggregation_failure(aggregate_metrics: Path) -> Optional[str]:
    """Why the aggregate metrics file is not a complete result, or None.

    A failed `/aggregate_metrics` request leaves an `aggregation_error` entry beside the healthy agents'
    metrics. A live run keeps those partial metrics and reports the failure, but an export marks the run
    done, so it must not send a partial result.
    """
    from nemo_gym.batch_status import AGGREGATION_ERROR_KEY

    try:
        entries = orjson.loads(aggregate_metrics.read_bytes())
    except orjson.JSONDecodeError as exc:
        return f"Aggregate metrics {aggregate_metrics} are not valid JSON ({exc}). Run `gym eval aggregate` again."
    failed = sorted(
        str((entry.get("agent_ref") or {}).get("name", "<unknown>"))
        for entry in entries
        if isinstance(entry, dict) and AGGREGATION_ERROR_KEY in entry
    )
    if failed:
        return f"Aggregation failed for agents: {', '.join(failed)}. Run `gym eval aggregate` again."
    return None


def unexportable_reason(artifacts: RunArtifacts, *, rollouts_required: bool = True) -> Optional[str]:
    """What is missing for the run to be exported faithfully, or None when it can be.

    Pass `rollouts_required=False` when the exporter does not read the rollouts file. When that file is there,
    the aggregate metrics must still not be older than it.
    """
    rollouts = artifacts.rollouts_jsonl_fpath
    if rollouts_required and not rollouts.is_file():
        return f"Rollouts file not found: {rollouts}"
    aggregate_metrics = artifacts.aggregate_metrics_json_fpath
    if not aggregate_metrics.is_file():
        return f"Aggregate metrics not found: {aggregate_metrics}. Run `gym eval aggregate` first."
    if rollouts.is_file() and aggregate_metrics.stat().st_mtime < rollouts.stat().st_mtime:
        return f"Aggregate metrics {aggregate_metrics} are older than {rollouts}. Run `gym eval aggregate` again."
    if not artifacts.resolved_config_yaml_fpath.is_file():
        return (
            f"Saved run config not found: {artifacts.resolved_config_yaml_fpath}. "
            "The run predates saving it; re-run it, or export it live."
        )
    return _aggregation_failure(aggregate_metrics)


class BaseExporter(Exporter):
    """Sink for the run metadata a NeMo Gym eval produces: config, metrics, rollouts.

    A backend (W&B, MLflow, ...) subclasses this and wraps its own run handle. Lifecycle is
    `setup` -> any number of `log_*` calls -> `teardown`. Whether a backend runs at all is decided
    by its `ExporterConfig` in the registry, not here.

    Exporters are best-effort telemetry: a failing tracking server must not fail the eval. Call
    sites go through the `export_*` wrappers, which swallow and log backend exceptions. Subclasses
    implement the unwrapped `_*` hooks and may raise freely.
    """

    # Every record-level exporter reads the run from the same rollouts file; the rest is its backend's settings.
    config_model: ClassVar[type[RunExportConfig]] = RunExportConfig

    def __init__(self, global_config_dict: DictConfig) -> None:
        # `config` is the subclass's own backend settings, validated from `global_config_dict`.
        self.global_config_dict = global_config_dict
        # Whether `setup` reopened the run recorded next to the output.
        self.reopened = False
        # Set by `start_over`: the output starts anew, so the recorded run is not reopened.
        self._starting_over = False
        # The run id `_recorded_run_id` read from the marker next to the output.
        self._recorded: Optional[str] = None
        # A reopened run's config, masked: logged once the run continues, dropped if the output starts over.
        self._pending_config: Optional[DictConfig] = None
        # Set only while `export()` is running, which is how the run-id hooks below tell a post-run export from a
        # live run (where it stays None). While it is set, the marker path comes from these artifacts instead of
        # `output_jsonl_fpath`, a recorded run id is not reused to resume, and the run id is held back in
        # `_pending_run_id` rather than written to the marker.
        self._exported_artifacts: Optional[RunArtifacts] = None
        # The backend run id reported during a post-run export. `export()` writes it to the marker once the whole
        # run has been sent.
        self._pending_run_id: Optional[str] = None

    @classmethod
    def from_global_config(cls, global_config_dict: DictConfig) -> "BaseExporter":
        """Build the exporter from the whole resolved config.

        This overrides `Exporter.from_global_config` because these exporters have always been built from the full
        config (a live run has no settings model of its own) instead of from a validated model. So only the shared
        settings are validated here, and each backend validates its own in its constructor.
        """
        cls.config_model.model_validate(global_config_dict)
        return cls(global_config_dict)

    @abstractmethod
    def setup(self) -> None:
        """Open the backing run. Called once, after config resolution."""

    @abstractmethod
    def teardown(self) -> None:
        """Close the backing run. Must be safe to call when `setup` failed or never ran."""

    @abstractmethod
    def _log_config(self, config_dict: DictConfig) -> None:
        """Record the resolved run config. Secrets are already masked by `export_config`."""

    @abstractmethod
    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Record scalar metrics. `step` is the training step, or None for a one-shot eval."""

    @abstractmethod
    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        """Record rollout results as a table. Rollouts are sanitized dicts, one per rollout."""

    def export_config(self) -> None:
        # `global_config_dict` holds live credentials (the backend needs its own API key to connect),
        # so mask a copy rather than shipping it to a tracking server as-is.
        config_dict_to_log = deepcopy(self.global_config_dict)
        recursively_hide_secrets(config_dict_to_log)
        if self.reopened:
            # Collection may still find nothing to resume and start the output over (`start_over`): the earlier
            # job's run must not get this job's config then.
            self._pending_config = config_dict_to_log
            return
        self._guard("config", self._log_config, config_dict_to_log)

    def export_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self._flush_pending_config()
        self._guard("metrics", self._log_metrics, metrics, step)

    def export_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        self._flush_pending_config()
        self._guard("rollouts", self._log_rollouts, rollouts)

    def _flush_pending_config(self) -> None:
        """Log the reopened run's config, as the run continues. Backends call this first in `teardown` too."""
        config_dict_to_log, self._pending_config = self._pending_config, None
        if config_dict_to_log is not None:
            self._guard("config", self._log_config, config_dict_to_log)

    def start_over(self) -> None:
        """Track the output in a new run after all.

        Rollout collection calls this when it was asked to resume its output but found nothing to resume and starts it
        anew, before it logs any rollout. A no-op unless `setup` reopened the recorded run.
        """
        if self.reopened:
            self._guard("a new run", self._start_over)

    def _start_over(self) -> None:
        self._pending_config = None
        try:
            self._close_reopened()
        except Exception as e:
            # Leaving the old run as it was is best effort; the output gets a run of its own either way.
            logger.warning(
                f"Exporter {self.name} could not restore the reopened run; starting a new one anyway: {e}",
                exc_info=True,
            )
        self._starting_over = True
        self._forget_recorded_run()
        self.setup()
        self.export_config()

    def _forget_recorded_run(self) -> None:
        """Remove the marker that names the reopened run, so that a failed replacement `setup` cannot leave the next
        resume reopening that run. A marker that names another run is left alone. Best effort: if the marker can be
        neither removed nor replaced by the new run's, the next resume still reopens the old run."""
        run_id_fpath = self._run_id_fpath()
        try:
            if run_id_fpath is not None and run_id_fpath.read_text().strip() == self._recorded:
                run_id_fpath.unlink()
        except OSError as e:
            logger.warning(
                f"Exporter {self.name} could not remove {run_id_fpath}, which names the run this output started over "
                f"from; unless the new run replaces it, the next resume reopens that run: {e}"
            )

    def _close_reopened(self) -> None:
        """Close the reopened run. A backend that marks a run when it reopens it undoes that mark here, and lets go
        of the run first, so that nothing more reaches it even when closing fails."""
        self.teardown()

    def _guard(self, what: str, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as e:
            logger.warning(f"Exporter {self.name} failed to log {what}; continuing: {e}", exc_info=True)

    def _run_id_fpath(self) -> Optional[Path]:
        """`<output stem>_<name>_run_id` next to `output_jsonl_fpath`, or None when the run has no output.

        A resumed or resubmitted run writes to the same output, so a backend that records its run id there
        reopens that run instead of starting a new one. Deleting the file starts a new run. Only a live run
        reads or writes it; a post-run export keeps its own marker (`RunArtifacts.export_done_fpath`).
        """
        output = self.global_config_dict.get("output_jsonl_fpath")
        if not output:
            return None
        return exporter_run_id_path_for(Path(output), self.name)

    def _recorded_run_id(self) -> Optional[str]:
        """The run to reopen: the one recorded next to the output, when this run continues that output.

        A resume continues it, as does a judge-failure recovery that appends to it. A fresh run into the same output
        starts its rollouts anew, so it starts a new tracking run too, as does a resume that finds nothing to resume
        (`start_over`).
        """
        if self._exported_artifacts is not None:
            return None
        run_id_fpath = self._run_id_fpath()
        continues = self._flag("resume_from_cache") or (self._flag("judge_failed_only") and self._flag("append"))
        if self._starting_over or run_id_fpath is None or not run_id_fpath.exists() or not continues:
            return None
        self._recorded = run_id_fpath.read_text().strip()
        return self._recorded

    def _flag(self, name: str) -> bool:
        """A flag of the raw config read as the validated run config reads it, so "false" or "0" (as an
        `${oc.env:...}` interpolation yields) is false. A value that config would reject does not continue a run."""
        value = self.global_config_dict.get(name)
        if value is None:
            return False
        try:
            return _BOOL.validate_python(value)
        except ValidationError:
            return False

    def _record_run_id(self, run_id: str) -> None:
        if self._exported_artifacts is not None:
            # Post-run exports leave the marker until the whole run has been sent.
            self._pending_run_id = run_id
            return
        run_id_fpath = self._run_id_fpath()
        if run_id_fpath is not None:
            run_id_fpath.parent.mkdir(parents=True, exist_ok=True)
            run_id_fpath.write_text(run_id)

    def export(self, artifacts: RunArtifacts) -> ExportResult:
        """Send a finished run to the backend: its saved config, aggregate metrics and (optionally) rollouts.

        Skipped, without contacting the backend, when the run was already exported. Failures come back in the
        result rather than being swallowed the way live `export_*` calls are.
        """
        marker = artifacts.export_done_fpath(self.name)
        if marker.exists():
            return ExportResult.skipped(reason="already exported", run_id=marker.read_text().strip())
        # The rollouts file is only required below, when the rollouts are actually uploaded.
        unexportable = unexportable_reason(artifacts, rollouts_required=False)
        if unexportable is not None:
            return ExportResult.failed(unexportable)

        live_run_id = _live_run_id(artifacts, self.name)
        if live_run_id is not None:
            # A live run's marker proves the backend run was opened, not that every upload landed, so it cannot
            # stand in for this export. Say so, as this run may now be in the backend twice.
            logger.warning(
                f"Exporter {self.name} logged {artifacts.rollouts_jsonl_fpath} live as run {live_run_id}; "
                "this export sends it again as a separate run."
            )
        self._exported_artifacts, self._pending_run_id = artifacts, None
        try:
            saved_config = OmegaConf.load(artifacts.resolved_config_yaml_fpath)
            upload = self._upload_enabled(saved_config)
            if upload and not artifacts.rollouts_jsonl_fpath.is_file():
                return ExportResult.failed(
                    f"Rollouts file not found: {artifacts.rollouts_jsonl_fpath}. "
                    "Pass `+upload_rollouts=false` to export without the rollouts."
                )
            metrics = flatten_aggregate_metrics(orjson.loads(artifacts.aggregate_metrics_json_fpath.read_bytes()))
            rollouts = self._read_rollouts(artifacts) if upload else None
            with self:
                self._log_config(saved_config)
                self._log_metrics(metrics)
                if rollouts is not None:
                    self._log_rollouts(rollouts)
            run_id = self._pending_run_id
            if not run_id:
                # The marker records which backend run holds this export, and a skipped repeat reports it. A backend
                # that never reports one is a bug in the exporter, and it shows on the first export.
                raise RuntimeError(
                    f"The {self.name} exporter did not report a run id: its setup() must call _record_run_id(). "
                    "The export is not marked as done."
                )
            _write_marker_atomically(marker, run_id)
        except Exception as exc:
            logger.warning(f"Exporter {self.name} failed to export {artifacts.rollouts_jsonl_fpath}", exc_info=True)
            return ExportResult.failed(f"{type(exc).__name__}: {exc}")
        finally:
            self._exported_artifacts = None
        detail = {"run_id": run_id, "uploaded_rollouts": rollouts is not None}
        if live_run_id is not None:
            detail["live_run_id"] = live_run_id
        return ExportResult.succeeded(**detail)

    def _upload_enabled(self, saved_config: DictConfig) -> bool:
        """Whether the rollouts are uploaded, as a live run would have decided.

        An explicit `upload_rollouts=` on the export command wins; otherwise the run's own saved setting does.
        """
        explicit = _upload_intent(self.global_config_dict)
        if "upload_rollouts" in explicit.model_fields_set:
            return explicit.upload_rollouts
        return _upload_intent(saved_config).upload_rollouts

    @staticmethod
    def _read_rollouts(artifacts: RunArtifacts) -> list[dict[str, Any]]:
        """The rollouts a live run would have uploaded."""
        from nemo_gym.rollout_collection import NG_DISPATCH_DRAINED_KEY, _rollout_for_export

        # Rows flagged as drained stand for tasks the dispatch budget never started, so there is nothing to upload,
        # and a live run leaves them out of its upload too. `_rollout_for_export` drops the bulky trajectory and
        # raw model-call payloads, the same trimming a live run applies.
        with artifacts.rollouts_jsonl_fpath.open("rb") as f:
            rows = (orjson.loads(line) for line in f if line.strip())
            return [_rollout_for_export(row) for row in rows if not row.get(NG_DISPATCH_DRAINED_KEY)]

    def __enter__(self) -> "BaseExporter":
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.teardown()
