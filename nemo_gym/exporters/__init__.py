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
"""Run-metadata exporters.

`setup_exporters` opens one exporter per configured backend during global config resolution; call
sites then fan out through the module-level `export_*` helpers.

A backend's module — and with it a multi-hundred-millisecond tracking SDK import — is loaded only
once its config says it is wired up. That is why the registry pairs each backend with a config
model from `config_types` rather than asking the exporter class whether it is available.
"""

import atexit
import logging
from importlib import import_module
from typing import Any, Optional

from omegaconf import DictConfig

from nemo_gym.config_types import ExporterConfig, MLFlowConfig, WANDBConfig
from nemo_gym.exporters.base import BaseExporter


logger = logging.getLogger(__name__)

# Backend name -> (config model, "module:class").
EXPORTER_REGISTRY: dict[str, tuple[type[ExporterConfig], str]] = {
    "wandb": (WANDBConfig, "nemo_gym.exporters.wandb:WandbExporter"),
    "mlflow": (MLFlowConfig, "nemo_gym.exporters.mlflow:MLflowExporter"),
}

_EXPORTERS: list[BaseExporter] = []


def _load_exporter_class(class_path: str) -> type[BaseExporter]:
    module_name, class_name = class_path.split(":")
    return getattr(import_module(module_name), class_name)


def get_exporters() -> list[BaseExporter]:
    """The exporters opened for this process. Empty when no backend is configured."""
    return list(_EXPORTERS)


def setup_exporters(global_config_dict: DictConfig) -> list[BaseExporter]:
    """Open every backend that is fully configured, and log the run config to each.

    Replaces any previously opened exporters. A backend that fails to start is skipped with a
    warning: telemetry must not take the run down with it.
    """
    teardown_exporters()

    for name, (config_model, class_path) in EXPORTER_REGISTRY.items():
        if not config_model.model_validate(global_config_dict).is_available:
            continue
        try:
            exporter = _load_exporter_class(class_path)(global_config_dict)
            exporter.setup()
        except Exception as e:
            logger.warning(f"Exporter {name} failed to start; continuing without it: {e}", exc_info=True)
            continue

        exporter.export_config()
        _EXPORTERS.append(exporter)

    # Registered here rather than at import so this lands after any hook a backend installed while
    # opening. atexit runs LIFO, so ours goes first: the wandb SDK closes its service in its own
    # hook, which would leave our teardown talking to a dead socket. Re-registering is harmless
    # because `teardown_exporters` is idempotent.
    atexit.register(teardown_exporters)
    return get_exporters()


def teardown_exporters() -> None:
    """Close all open exporters. Safe to call repeatedly and when none are open."""
    while _EXPORTERS:
        exporter = _EXPORTERS.pop()
        try:
            exporter.teardown()
        except Exception as e:
            logger.warning(f"Exporter {exporter.name} failed to shut down cleanly: {e}", exc_info=True)


def export_metrics(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    for exporter in _EXPORTERS:
        exporter.export_metrics(metrics, step)


def export_rollouts(rollouts: list[dict[str, Any]]) -> None:
    for exporter in _EXPORTERS:
        exporter.export_rollouts(rollouts)
