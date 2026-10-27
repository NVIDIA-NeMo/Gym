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
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Optional

from omegaconf import DictConfig
from pydantic import TypeAdapter, ValidationError

from nemo_gym.secret_utils import recursively_hide_secrets


logger = logging.getLogger(__name__)
# Pydantic's boolean parsing, which the run configs that read these flags use.
_BOOL = TypeAdapter(bool)


class BaseExporter(ABC):
    """Sink for the run metadata a NeMo Gym eval produces: config, metrics, rollouts.

    A backend (W&B, MLflow, ...) subclasses this and wraps its own run handle. Lifecycle is
    `setup` -> any number of `log_*` calls -> `teardown`. Whether a backend runs at all is decided
    by its `ExporterConfig` in the registry, not here.

    Exporters are best-effort telemetry: a failing tracking server must not fail the eval. Call
    sites go through the `export_*` wrappers, which swallow and log backend exceptions. Subclasses
    implement the unwrapped `_*` hooks and may raise freely.
    """

    # Config key prefix and identifier used in logs, e.g. "wandb".
    name: ClassVar[str]

    def __init__(self, global_config_dict: DictConfig) -> None:
        self.global_config_dict = global_config_dict
        # Whether `setup` reopened the run recorded next to the output.
        self.reopened = False
        # Set by `start_over`: the output starts anew, so the recorded run is not reopened.
        self._starting_over = False
        # The run id `_recorded_run_id` read from the marker next to the output.
        self._recorded: Optional[str] = None
        # A reopened run's config, masked: logged once the run continues, dropped if the output starts over.
        self._pending_config: Optional[DictConfig] = None

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
        reopens that run instead of starting a new one. Deleting the file starts a new run.
        """
        output = self.global_config_dict.get("output_jsonl_fpath")
        if not output:
            return None
        output = Path(output)
        return output.with_name(f"{output.stem}_{self.name}_run_id")

    def _recorded_run_id(self) -> Optional[str]:
        """The run to reopen: the one recorded next to the output, when this run continues that output.

        A resume continues it, as does a judge-failure recovery that appends to it. A fresh run into the same output
        starts its rollouts anew, so it starts a new tracking run too, as does a resume that finds nothing to resume
        (`start_over`).
        """
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
        run_id_fpath = self._run_id_fpath()
        if run_id_fpath is not None:
            run_id_fpath.parent.mkdir(parents=True, exist_ok=True)
            run_id_fpath.write_text(run_id)

    def __enter__(self) -> "BaseExporter":
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.teardown()
