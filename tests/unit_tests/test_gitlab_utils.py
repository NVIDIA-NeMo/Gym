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
from nemo_gym.config_types import MLFlowConfig


# TODO: Eventually we want to add more tests to ensure that the Gitlab flow does not break
class TestGitlabUtils:
    def test_sanity(self) -> None:
        MLFlowConfig(mlflow_tracking_uri="", mlflow_tracking_token="")

    def test_registry_credentials_alone_do_not_enable_the_exporter(self) -> None:
        config = MLFlowConfig(mlflow_tracking_uri="https://gitlab", mlflow_tracking_token="t")

        assert not config.is_available
