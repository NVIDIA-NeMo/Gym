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
from typing import Any, ClassVar, Optional

from omegaconf import DictConfig

from nemo_gym.secret_utils import recursively_hide_secrets


logger = logging.getLogger(__name__)


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
        self._guard("config", self._log_config, config_dict_to_log)

    def export_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self._guard("metrics", self._log_metrics, metrics, step)

    def export_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        self._guard("rollouts", self._log_rollouts, rollouts)

    def _guard(self, what: str, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as e:
            logger.warning(f"Exporter {self.name} failed to log {what}; continuing: {e}", exc_info=True)

    def __enter__(self) -> "BaseExporter":
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.teardown()
