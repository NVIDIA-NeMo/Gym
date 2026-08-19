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
from math import isfinite
from os import environ
from pathlib import Path
from time import time
from typing import Any, ClassVar, Iterator, Optional

import orjson
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.utils.validation import (
    MAX_ENTITY_KEY_LENGTH,
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
)
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config_types import MLFlowConfig
from nemo_gym.exporters.base import BaseExporter


logger = logging.getLogger(__name__)


def _flatten_config(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (dotted_key, leaf) pairs. MLflow params are flat, unlike W&B's nested config."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _flatten_config(inner, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            yield from _flatten_config(inner, f"{prefix}[{index}]")
    elif prefix:
        yield prefix, value


def _chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class MLflowExporter(BaseExporter):
    """MLflow backend.

    Configured by `mlflow_tracking_uri`, `mlflow_tracking_token`, `mlflow_experiment_name` and
    `mlflow_run_name` in the global config.

    Uses `MlflowClient` with an explicit run id rather than the fluent `mlflow.*` API, which
    tracks the active run in thread-local state that Gym's async call sites don't share.
    """

    name: ClassVar[str] = "mlflow"

    CONFIG_ARTIFACT_FILE: ClassVar[str] = "global_config.json"
    ROLLOUTS_ARTIFACT_FILE: ClassVar[str] = "rollouts.json"

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.config = MLFlowConfig.model_validate(global_config_dict)
        self.client: Optional[MlflowClient] = None
        self.run_id: Optional[str] = None

    def setup(self) -> None:
        # The MLflow SDK reads the bearer token from the environment, not from client kwargs.
        environ["MLFLOW_TRACKING_TOKEN"] = self.config.mlflow_tracking_token
        self.client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
        experiment_id = self._experiment_id(self.config.mlflow_experiment_name)
        self.run_id = self.client.create_run(experiment_id, run_name=self.config.mlflow_run_name).info.run_id

    def teardown(self) -> None:
        if self.client is not None and self.run_id is not None:
            self.client.set_terminated(self.run_id, status="FINISHED")
        self.client = None
        self.run_id = None

    def _experiment_id(self, name: str) -> str:
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self.client.create_experiment(name)
        except MlflowException:
            # Lost a create race against a concurrent shard; the experiment now exists.
            return self.client.get_experiment_by_name(name).experiment_id

    def _active(self) -> tuple[MlflowClient, str]:
        if self.client is None or self.run_id is None:
            raise RuntimeError("MLflow run is not open; call setup() before logging.")
        return self.client, self.run_id

    def _log_config(self, config_dict: DictConfig) -> None:
        client, run_id = self._active()
        container = OmegaConf.to_container(config_dict)

        # Log the whole config first: params are lossy (flat, truncated, length-capped), so the
        # artifact is the record of what actually ran.
        client.log_dict(run_id, container, self.CONFIG_ARTIFACT_FILE)

        params = [
            Param(key, str(value)[:MAX_PARAM_VAL_LENGTH])
            for key, value in _flatten_config(container)
            if len(key) <= MAX_ENTITY_KEY_LENGTH
        ]
        for batch in _chunked(params, MAX_PARAMS_TAGS_PER_BATCH):
            client.log_batch(run_id, params=batch)

    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        client, run_id = self._active()
        timestamp = int(time() * 1000)

        # MLflow metrics must be numeric and finite. Strings become tags so they aren't lost;
        # None and NaN/inf (e.g. the std of a single-rollout task) are dropped.
        numeric: list[Metric] = []
        tags: list[RunTag] = []
        for key, value in metrics.items():
            if len(key) > MAX_ENTITY_KEY_LENGTH:
                continue
            if isinstance(value, bool) or isinstance(value, (int, float)):
                if isfinite(value):
                    numeric.append(Metric(key, float(value), timestamp, step or 0))
            elif isinstance(value, str):
                tags.append(RunTag(key, value[:MAX_PARAM_VAL_LENGTH]))

        for batch in _chunked(numeric, MAX_METRICS_PER_BATCH):
            client.log_batch(run_id, metrics=batch)
        for batch in _chunked(tags, MAX_PARAMS_TAGS_PER_BATCH):
            client.log_batch(run_id, tags=batch)

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        client, run_id = self._active()
        # One JSON blob per row, matching the W&B rollouts table: rollouts have no stable column
        # set across environments. `log_table` appends when the artifact already exists.
        rows = [orjson.dumps(rollout).decode() for rollout in rollouts]
        client.log_table(run_id, data={"Rollout": rows}, artifact_file=self.ROLLOUTS_ARTIFACT_FILE)

    def _log_artifacts(self, artifacts_dirpath: Path) -> None:
        client, run_id = self._active()
        client.log_artifacts(run_id, str(artifacts_dirpath))
