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
from typing import Any, ClassVar, Optional

from langsmith import Client
from omegaconf import DictConfig

from nemo_gym.config_types import LangSmithConfig
from nemo_gym.exporters.base import BaseExporter


class LangSmithExporter(BaseExporter):
    """Export Gym evaluation results to LangSmith."""

    name: ClassVar[str] = "langsmith"

    def __init__(self, global_config_dict: DictConfig) -> None:
        super().__init__(global_config_dict)
        self.config = LangSmithConfig.model_validate(global_config_dict)
        self.client: Optional[Client] = None

    def setup(self) -> None:
        self.client = Client(
            api_url=self.config.langsmith_endpoint,
            api_key=self.config.langsmith_api_key,
            workspace_id=self.config.langsmith_workspace_id,
        )

    def teardown(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def _log_config(self, _config_dict: DictConfig) -> None:
        pass

    def _log_metrics(self, _metrics: dict[str, Any], _step: Optional[int] = None) -> None:
        pass

    def _log_rollouts(self, _rollouts: list[dict[str, Any]]) -> None:
        pass
