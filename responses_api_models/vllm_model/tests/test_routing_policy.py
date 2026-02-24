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

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from responses_api_models.vllm_model.routing_policy import (
    CacheAwareRoutingPolicy,
    CacheAwareRoutingPolicyConfig,
    RoundRobinRoutingPolicy,
    RoundRobinRoutingPolicyConfig,
    create_routing_policy,
)

DUMMY_REQUEST_BODY = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}


def _make_mock_clients(n: int):
    return [MagicMock(spec=NeMoGymAsyncOpenAI) for _ in range(n)]


class TestRoundRobinRoutingPolicy:
    def test_round_robin_distribution(self):
        policy = RoundRobinRoutingPolicy(_make_mock_clients(3))
        results = [policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id=f"req_{i}") for i in range(6)]
        assert results == [0, 1, 2, 0, 1, 2]

    def test_session_stickiness(self):
        policy = RoundRobinRoutingPolicy(_make_mock_clients(3))
        # First request for session_a goes to client 0
        idx_a1 = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_1", session_id="session_a")
        assert idx_a1 == 0

        # First request for session_b goes to client 1 (round-robin)
        idx_b1 = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_2", session_id="session_b")
        assert idx_b1 == 1

        # Second request for session_a returns same client 0 (sticky)
        idx_a2 = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_3", session_id="session_a")
        assert idx_a2 == 0

        # Second request for session_b returns same client 1 (sticky)
        idx_b2 = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_4", session_id="session_b")
        assert idx_b2 == 1

    def test_no_session_id_always_round_robins(self):
        policy = RoundRobinRoutingPolicy(_make_mock_clients(2))
        results = [policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id=f"req_{i}") for i in range(4)]
        assert results == [0, 1, 0, 1]

    def test_lifecycle_callbacks_are_noop(self):
        policy = RoundRobinRoutingPolicy(_make_mock_clients(1))
        # These should not raise
        policy.on_prefill_complete("req_1")
        policy.on_generation_complete("req_1")


class TestCacheAwareRoutingPolicy:
    def _make_policy(self, num_clients=3, route_return=0):
        mock_router = MagicMock()
        mock_router.route.return_value = route_return
        clients = _make_mock_clients(num_clients)
        policy = CacheAwareRoutingPolicy(mock_router, clients)
        return policy, mock_router

    def test_delegates_to_external_router(self):
        policy, mock_router = self._make_policy(num_clients=3, route_return=1)
        idx = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_1")
        assert idx == 1
        mock_router.route.assert_called_once_with(DUMMY_REQUEST_BODY)

    def test_fallback_on_out_of_range_index(self):
        # External router returns index 5 but only 3 clients
        policy, _ = self._make_policy(num_clients=3, route_return=5)
        idx = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_1")
        assert idx == 5 % 3  # 2

    def test_negative_index_fallback(self):
        policy, _ = self._make_policy(num_clients=3, route_return=-1)
        idx = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_1")
        assert 0 <= idx < 3

    def test_valid_index_no_fallback(self):
        policy, _ = self._make_policy(num_clients=3, route_return=0)
        idx = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_1")
        assert idx == 0

    def test_on_prefill_complete(self):
        policy, mock_router = self._make_policy()
        policy.on_prefill_complete("req_42")
        mock_router.prefill_complete.assert_called_once_with("req_42")

    def test_on_generation_complete(self):
        policy, mock_router = self._make_policy()
        policy.on_generation_complete("req_42")
        mock_router.generation_complete.assert_called_once_with("req_42")


class TestCreateRoutingPolicy:
    def test_creates_round_robin(self):
        clients = _make_mock_clients(2)
        config = RoundRobinRoutingPolicyConfig()
        policy = create_routing_policy(config, clients)
        assert isinstance(policy, RoundRobinRoutingPolicy)

    def test_creates_cache_aware(self):
        clients = _make_mock_clients(2)
        config = CacheAwareRoutingPolicyConfig(
            router_class="unittest.mock.MagicMock",
            router_kwargs={"name": "test_router"},
        )
        policy = create_routing_policy(config, clients)
        assert isinstance(policy, CacheAwareRoutingPolicy)

    def test_cache_aware_passes_kwargs(self):
        clients = _make_mock_clients(2)
        config = CacheAwareRoutingPolicyConfig(
            router_class="unittest.mock.MagicMock",
            router_kwargs={"some_param": 42},
        )
        policy = create_routing_policy(config, clients)
        assert isinstance(policy, CacheAwareRoutingPolicy)

    def test_cache_aware_invalid_class_raises(self):
        clients = _make_mock_clients(2)
        config = CacheAwareRoutingPolicyConfig(
            router_class="nonexistent.module.ClassName",
        )
        with pytest.raises(ModuleNotFoundError):
            create_routing_policy(config, clients)


class TestRoutingPolicyConfig:
    def test_round_robin_config_defaults(self):
        config = RoundRobinRoutingPolicyConfig()
        assert config.type == "round_robin"

    def test_cache_aware_config(self):
        config = CacheAwareRoutingPolicyConfig(
            router_class="my_module.MyRouter",
            router_kwargs={"key": "value"},
        )
        assert config.type == "cache_aware"
        assert config.router_class == "my_module.MyRouter"
        assert config.router_kwargs == {"key": "value"}

    def test_cache_aware_config_default_kwargs(self):
        config = CacheAwareRoutingPolicyConfig(
            router_class="my_module.MyRouter",
        )
        assert config.router_kwargs == {}
