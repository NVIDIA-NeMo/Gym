# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light, environment-owned dataset metric hooks."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias


DatasetMetricScalar: TypeAlias = bool | int | float | str
DatasetMetricValue: TypeAlias = DatasetMetricScalar | list[DatasetMetricScalar] | None
DatasetMetricHook: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, DatasetMetricValue]]

DATASET_METRICS_MODULE_NAME = "dataset_metrics"
DATASET_METRICS_EXPORT_NAME = "compute_task_metrics"


class DatasetMetricsHookError(Exception):
    """An environment's dataset metrics hook does not satisfy the protocol."""


def load_dataset_metrics_hook(server_dir: Path) -> DatasetMetricHook | None:
    """Load an optional ``compute_task_metrics`` hook from an owning server directory."""
    module_path = server_dir / f"{DATASET_METRICS_MODULE_NAME}.py"
    if not module_path.is_file():
        return None

    module_name = f"nemo_gym_dataset_metrics.{server_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib internals
        raise DatasetMetricsHookError(f"Could not build an import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise DatasetMetricsHookError(f"Failed to import {module_path}: {error}") from error

    hook = getattr(module, DATASET_METRICS_EXPORT_NAME, None)
    if hook is None:
        raise DatasetMetricsHookError(f"{module_path} does not export `{DATASET_METRICS_EXPORT_NAME}(task_input)`.")
    if not callable(hook):
        raise DatasetMetricsHookError(f"`{DATASET_METRICS_EXPORT_NAME}` in {module_path} is not callable.")
    return hook
