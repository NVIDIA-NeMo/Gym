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
from unittest.mock import MagicMock

from app import ExampleMultiStepResourcesServer, ExampleMultiStepResourcesServerConfig

from nemo_gym.server_utils import ServerClient


def _make_config() -> ExampleMultiStepResourcesServerConfig:
    return ExampleMultiStepResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_responses_create_params={"model": "judge_model", "input": []},
    )


class TestApp:
    def test_sanity(self) -> None:
        ExampleMultiStepResourcesServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))

    def test_parse_reward_reads_leading_number(self) -> None:
        server = ExampleMultiStepResourcesServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        judge_response = MagicMock()
        judge_response.output[-1].content[-1].text = "0.75 — mostly correct"
        assert server._parse_reward(judge_response) == 0.75

    def test_parse_reward_falls_back_to_zero_on_unparseable_reply(self) -> None:
        server = ExampleMultiStepResourcesServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))

        judge_response = MagicMock()
        judge_response.output[-1].content[-1].text = "no score here"
        assert server._parse_reward(judge_response) == 0.0
