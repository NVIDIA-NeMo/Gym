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

from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.failure_kinds import JUDGE_FAILED, SESSION_LOST
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract


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
    def test_run_request_does_not_emit_environment_specific_control_fields(self) -> None:
        request = BaseRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
        )

        assert set(request.model_dump()) == {"responses_create_params"}

    def test_sanity(self) -> None:
        _resources_server().setup_webserver()

    def test_reverify_mode(self) -> None:
        assert asyncio.run(_resources_server().get_reverify_mode()) == ReverifyMode.UNKNOWN

    def test_stateless_server_answers_typed_and_legacy_session_calls(self) -> None:
        """A server with no per-rollout state serves an Environment Server's seed and close without overrides."""
        client = TestClient(_resources_server().setup_webserver())
        identity = {"episode_id": {"rollout_id": "rollout", "attempt": 0}}

        typed_seed = client.post(
            "/seed_session",
            json={
                "resources_session_id": "resources-session",
                "task_id": {"taskset": "tasks", "task_id": "task"},
                "task_data": {"question": "2+2"},
            }
            | identity,
        )
        # An Agent's /run seeds with its legacy row, which keeps the empty response.
        legacy_seed = client.post("/seed_session", json={"responses_create_params": {"input": "2+2"}})
        typed_close = client.post("/close_session", json={"resources_session_id": "resources-session"} | identity)

        assert typed_seed.status_code == 200
        assert typed_seed.json()["resources_session_id"] == "resources-session"
        assert (legacy_seed.status_code, legacy_seed.json()) == (200, {})
        assert (typed_close.status_code, typed_close.json()) == (200, {"resources_session_id": "resources-session"})

    def test_stateless_server_follows_the_session_contract(self) -> None:
        check_resources_session_contract(
            _resources_server().setup_webserver(),
            ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="tasks", task_id="task"),
                task_data={},
            ),
            keeps_state=False,
        )


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


class TestFailureKindOnTheContract:
    """The groupable half of the diagnosis, drawn from the shared vocabulary."""

    def _params(self) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(input="hi")

    def _response(self) -> NeMoGymResponse:
        return NeMoGymResponse.model_construct(id="resp-1", output=[])

    def _verify(self, **kwargs) -> BaseVerifyResponse:
        return BaseVerifyResponse(
            responses_create_params=self._params(), response=self._response(), reward=0.0, **kwargs
        )

    def test_absent_by_default(self) -> None:
        assert self._verify().failure_kind is None

    def test_a_registered_kind_round_trips(self) -> None:
        assert self._verify(failure_kind=SESSION_LOST).model_dump()["failure_kind"] == SESSION_LOST

    def test_naming_a_kind_does_not_decide_usability(self) -> None:
        """A degraded but validly measured sample names a kind and stays unmasked."""
        degraded = self._verify(failure_kind=JUDGE_FAILED, failure_reason="one retry was needed")

        assert degraded.mask_sample is False
        assert degraded.failure_kind == JUDGE_FAILED

    def test_an_unregistered_kind_warns_but_the_response_survives(self, caplog) -> None:
        """Dropping the response would replace a wrong label with a lost failure."""
        import logging

        from nemo_gym import failure_kinds

        failure_kinds._WARNED_UNKNOWN.discard("invented_kind")
        with caplog.at_level(logging.WARNING):
            response = self._verify(failure_kind="invented_kind")

        assert response.failure_kind == "invented_kind"
        assert any("invented_kind" in record.getMessage() for record in caplog.records)

    def test_an_environment_can_namespace_its_own(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            response = self._verify(failure_kind="lexmount_browser:quota_exhausted")

        assert response.failure_kind == "lexmount_browser:quota_exhausted"
        assert caplog.records == []
