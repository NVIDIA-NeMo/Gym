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
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from responses_api_models.vllm_model.routing_policy import (
    CacheAwareRoutingPolicyConfig,
    RoundRobinRoutingPolicy,
    RoundRobinRoutingPolicyConfig,
    RoutingPolicy,
    create_routing_policy,
)

DUMMY_REQUEST_BODY = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}


def _make_mock_clients(n: int) -> List[NeMoGymAsyncOpenAI]:
    return [MagicMock(spec=NeMoGymAsyncOpenAI) for _ in range(n)]


# ---------------------------------------------------------------------------
# Test stub: a minimal RoutingPolicy subclass usable as an external router
# ---------------------------------------------------------------------------


class _StubCacheAwarePolicy(RoutingPolicy):
    """Minimal RoutingPolicy subclass for testing the factory and lifecycle hooks."""

    def __init__(self, clients: List[NeMoGymAsyncOpenAI], route_return: int = 0, **kwargs):
        self._clients = clients
        self._route_return = route_return
        self.route_calls: List[Dict[str, Any]] = []
        self.prefill_calls: List[str] = []
        self.generation_calls: List[str] = []

    def select_client(
        self,
        *,
        request_body: Dict[str, Any],
        request_id: str,
        session_id: Optional[str] = None,
    ) -> int:
        self.route_calls.append(request_body)
        return self._route_return

    def on_prefill_complete(self, request_id: str) -> None:
        self.prefill_calls.append(request_id)

    def on_generation_complete(self, request_id: str) -> None:
        self.generation_calls.append(request_id)


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


class TestExternalRoutingPolicy:
    """Tests for external routing policies that subclass RoutingPolicy directly."""

    def test_stub_receives_request_body(self):
        clients = _make_mock_clients(2)
        policy = _StubCacheAwarePolicy(clients, route_return=1)
        idx = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_1")
        assert idx == 1
        assert policy.route_calls == [DUMMY_REQUEST_BODY]

    def test_stub_lifecycle_callbacks(self):
        clients = _make_mock_clients(2)
        policy = _StubCacheAwarePolicy(clients)
        policy.on_prefill_complete("req_42")
        policy.on_generation_complete("req_42")
        assert policy.prefill_calls == ["req_42"]
        assert policy.generation_calls == ["req_42"]


class TestCreateRoutingPolicy:
    def test_creates_round_robin(self):
        clients = _make_mock_clients(2)
        config = RoundRobinRoutingPolicyConfig()
        policy = create_routing_policy(config, clients)
        assert isinstance(policy, RoundRobinRoutingPolicy)

    def test_creates_external_policy(self):
        clients = _make_mock_clients(2)
        config = CacheAwareRoutingPolicyConfig(
            router_class="responses_api_models.vllm_model.tests.test_routing_policy._StubCacheAwarePolicy",
        )
        policy = create_routing_policy(config, clients)
        assert isinstance(policy, RoutingPolicy)
        assert isinstance(policy, _StubCacheAwarePolicy)

    def test_external_policy_receives_clients_and_kwargs(self):
        clients = _make_mock_clients(3)
        config = CacheAwareRoutingPolicyConfig(
            router_class="responses_api_models.vllm_model.tests.test_routing_policy._StubCacheAwarePolicy",
            router_kwargs={"route_return": 2},
        )
        policy = create_routing_policy(config, clients)
        assert isinstance(policy, _StubCacheAwarePolicy)
        assert policy._clients is clients
        idx = policy.select_client(request_body=DUMMY_REQUEST_BODY, request_id="req_1")
        assert idx == 2

    def test_non_routing_policy_class_raises(self):
        """Classes that don't subclass RoutingPolicy are rejected."""
        clients = _make_mock_clients(2)
        config = CacheAwareRoutingPolicyConfig(
            router_class="builtins.object",
        )
        with pytest.raises(TypeError, match="must be a subclass of RoutingPolicy"):
            create_routing_policy(config, clients)

    def test_invalid_module_raises(self):
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
