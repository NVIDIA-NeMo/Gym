# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from hypotest.dataset_server import HypotestDataset, HypotestDatasetConfig
from hypotest.env.interpreter_env import InterpreterEnv
from pydantic import model_validator

from nemo_gym.resources_servers.aviary.app import AviaryResourcesServer
from nemo_gym.resources_servers.aviary.schemas import AviaryResourcesServerConfig


class HypotestServerConfig(AviaryResourcesServerConfig):
    # dataset config
    dataset: HypotestDatasetConfig


class HypotestResourcesServer(AviaryResourcesServer[InterpreterEnv, HypotestDataset]):
    config: HypotestServerConfig
    dataset: HypotestDataset

    @model_validator(mode="before")
    @classmethod
    def load_dataset(cls, data: dict) -> dict:
        if "dataset" not in data:
            config = data["config"] = HypotestServerConfig.model_validate(data.get("config", {}))
            data["dataset"] = HypotestDataset(config.dataset)
        return data


if __name__ == "__main__":
    HypotestResourcesServer.run_webserver()
