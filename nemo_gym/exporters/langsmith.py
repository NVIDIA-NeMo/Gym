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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

from langsmith import Client
from omegaconf import DictConfig

from nemo_gym.config_types import LangSmithConfig
from nemo_gym.exporters.base import BaseExporter


_LANGSMITH_RUNNER_METADATA = {
    "ls_runner": "nemo-gym",
    "source": "nemo-gym",
}

_ROLLOUT_PAYLOAD_KEYS = frozenset(
    {
        "responses_create_params",
        "response",
        "reward",
    }
)


@dataclass(frozen=True)
class _MappedRollout:
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]
    reward: float


def _map_rollout(rollout: dict[str, Any]) -> _MappedRollout:
    metadata = {key: value for key, value in rollout.items() if key not in _ROLLOUT_PAYLOAD_KEYS}
    metadata.update(_LANGSMITH_RUNNER_METADATA)

    return _MappedRollout(
        inputs=rollout["responses_create_params"],
        outputs=rollout["response"],
        metadata=metadata,
        reward=float(rollout["reward"]),
    )


class LangSmithExporter(BaseExporter):
    """Export Gym evaluation results to LangSmith."""

    name: ClassVar[str] = "langsmith"

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.config = LangSmithConfig.model_validate(global_config_dict)
        self.client: Optional[Client] = None
        self.dataset_id: Optional[str] = None
        self.experiment_id: Optional[str] = None

    def setup(self) -> None:
        self.client = Client(
            api_url=self.config.langsmith_endpoint,
            api_key=self.config.langsmith_api_key,
            workspace_id=self.config.langsmith_workspace_id,
        )
        dataset_name = self.config.langsmith_dataset_name

        if self.client.has_dataset(dataset_name=dataset_name):
            dataset = self.client.read_dataset(dataset_name=dataset_name)
        else:
            dataset = self.client.create_dataset(
                dataset_name,
                description="Evaluation dataset exported by NeMo Gym.",
                metadata=_LANGSMITH_RUNNER_METADATA,
            )

        self.dataset_id = str(dataset.id)

        # LangSmith's SDK calls evaluation experiments projects.
        experiment = self.client.create_project(
            self.config.langsmith_experiment_name,
            upsert=True,
            reference_dataset_id=self.dataset_id,
            metadata=_LANGSMITH_RUNNER_METADATA,
        )
        self.experiment_id = str(experiment.id)

    def teardown(self) -> None:
        if self.client is None:
            return

        client = self.client

        try:
            if self.experiment_id is not None:
                client.update_project(
                    self.experiment_id,
                    end_time=datetime.now(timezone.utc),
                )
        finally:
            try:
                client.close()
            finally:
                self.client = None
                self.dataset_id = None
                self.experiment_id = None

    def _log_config(self, _config_dict: DictConfig) -> None:
        pass

    def _log_metrics(self, _metrics: dict[str, Any], _step: Optional[int] = None) -> None:
        pass

    def _log_rollouts(self, _rollouts: list[dict[str, Any]]) -> None:
        pass
