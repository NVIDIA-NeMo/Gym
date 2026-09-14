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
from os import environ
from pathlib import Path
from typing import Any, ClassVar, Optional

import orjson
import wandb
import wandb.util
from omegaconf import DictConfig, OmegaConf
from wandb import Run, Table

from nemo_gym.config_types import WANDBConfig
from nemo_gym.exporters.base import BaseExporter
from nemo_gym.global_config import RESULTS_DIR_KEY_NAME


# Increase row limit since some of our rollouts are pretty hefty
wandb.util.VALUE_BYTES_LIMIT = 10_000_000


class WandbExporter(BaseExporter):
    """Weights & Biases backend.

    Configured by `wandb_project`, `wandb_name` and `wandb_api_key` in the global config.
    """

    name: ClassVar[str] = "wandb"

    ROLLOUTS_TABLE_KEY: ClassVar[str] = "Rollouts"

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.config = WANDBConfig.model_validate(global_config_dict)
        self.run: Optional[Run] = None

    def setup(self) -> None:
        # The wandb SDK reads the key from the environment rather than from init kwargs.
        environ["WANDB_API_KEY"] = self.config.wandb_api_key
        self.run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            dir=str(Path(self.global_config_dict[RESULTS_DIR_KEY_NAME]) / "wandb"),
        )

    def teardown(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None

    def _active_run(self) -> Run:
        if self.run is None:
            raise RuntimeError("W&B run is not open; call setup() before logging.")
        return self.run

    def _log_config(self, config_dict: DictConfig) -> None:
        self._active_run().config.update(OmegaConf.to_container(config_dict))

    def _log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        # @bxyu-nvidia: Commit here so the rollouts show up in W&B on the current step, rather than being flushed in the next step
        self._active_run().log(metrics, step=step, commit=True)

    def _log_rollouts(self, rollouts: list[dict[str, Any]]) -> None:
        # One JSON blob per row: rollouts have no stable column set across environments.
        rows = [[orjson.dumps(rollout)] for rollout in rollouts]
        self._active_run().log({self.ROLLOUTS_TABLE_KEY: Table(data=rows, columns=["Rollout"])})
