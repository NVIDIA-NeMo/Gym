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
import asyncio
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


def _resources_server() -> SimpleResourcesServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

    class TestSimpleResourcesServer(SimpleResourcesServer):
        async def verify(self, body):
            pass

    return TestSimpleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestBaseVerifyResponse:
    def test_failure_reason_defaults_none_and_round_trips(self) -> None:
        response = BaseVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=0.0,
        )
        assert response.failure_reason is None
        assert response.model_dump()["failure_reason"] is None

        rescued = response.model_copy(update={"failure_reason": "judge response unparseable after 3 attempts"})
        assert rescued.model_dump()["failure_reason"] == "judge response unparseable after 3 attempts"
        assert rescued.reward == 0.0


class TestBaseMultiRewardVerifyResponse:
    def test_reward_components_round_trip(self) -> None:
        response = BaseMultiRewardVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=2.0,
            reward_components={"correctness": 1.0, "format": 1.0},
        )
        dumped = response.model_dump()
        assert dumped["reward_components"] == {"correctness": 1.0, "format": 1.0}
        assert dumped["reward"] == 2.0


class TestBaseResourcesServer:
    def test_sanity(self) -> None:
        _resources_server().setup_webserver()

    def test_reverify_mode(self) -> None:
        assert asyncio.run(_resources_server().get_reverify_mode()) == ReverifyMode.UNKNOWN


class TestVerifyResponseFailureReporting:
    """`mask_sample` / `failure_reason` on the contract, so every environment can use them."""

    def _params(self) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(input="hi")

    def _response(self) -> NeMoGymResponse:
        return NeMoGymResponse.model_construct(id="resp-1", output=[])

    def test_defaults_keep_existing_environments_unchanged(self) -> None:
        response = BaseVerifyResponse(responses_create_params=self._params(), response=self._response(), reward=0.0)
        assert response.mask_sample is False
        assert response.failure_reason is None

    def test_round_trip_preserves_the_flag_and_reason(self) -> None:
        response = BaseVerifyResponse(
            responses_create_params=self._params(),
            response=self._response(),
            reward=0.0,
            mask_sample=True,
            failure_reason="judge_unavailable",
        )
        dumped = response.model_dump()
        assert dumped["mask_sample"] is True
        assert dumped["failure_reason"] == "judge_unavailable"

    def test_zero_reward_is_not_implicitly_masked(self) -> None:
        """A policy that scores zero must stay a valid sample."""
        response = BaseVerifyResponse(responses_create_params=self._params(), response=self._response(), reward=0.0)
        assert response.mask_sample is False

    def test_a_degraded_but_legitimately_scored_rollout_is_not_masked(self) -> None:
        """The diagnosis is independent of the decision to exclude the sample."""
        response = BaseVerifyResponse(
            responses_create_params=self._params(),
            response=self._response(),
            reward=0.4,
            failure_reason="one retry was needed to reach the judge",
        )
        assert response.mask_sample is False
        assert response.failure_reason is not None
