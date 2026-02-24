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
import importlib
from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, Literal, Optional, Union, List

from pydantic import BaseModel, Field
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


class RoutingPolicy(ABC):
    @abstractmethod
    def select_client(
        self,
        *,
        request_body: Dict[str, Any],
        request_id: str,
        session_id: Optional[str] = None,
    ) -> int:
        """Select a client index for routing a request.

        Args:
            request_body: The raw chat completion request body dict.
            request_id: Unique identifier for this request.
            session_id: Optional session identifier for sticky routing.

        Returns:
            Index of the client to route the request to.
        """

    def on_prefill_complete(self, request_id: str) -> None:
        """Called when prefill is complete for a request. Default no-op."""

    def on_generation_complete(self, request_id: str) -> None:
        """Called when generation is complete for a request. Default no-op."""


class RoundRobinRoutingPolicy(RoutingPolicy):
    """Round-robin routing with session stickiness.

    If a session_id has been seen before, the same client is returned.
    Otherwise, clients are assigned in round-robin order.
    """

    def __init__(self, clients: List[NeMoGymAsyncOpenAI]):
        self._clients = clients
        self._session_id_to_client: Dict[str, int] = {}
        self._counter: int = 0

    def select_client(
        self,
        *,
        request_body: Dict[str, Any],
        request_id: str,
        session_id: Optional[str] = None,
    ) -> int:
        if session_id is not None and session_id in self._session_id_to_client:
            return self._session_id_to_client[session_id]

        client_idx = self._counter % len(self._clients)
        self._counter += 1

        if session_id is not None:
            self._session_id_to_client[session_id] = client_idx

        return client_idx


class CacheAwareRoutingPolicy(RoutingPolicy):
    """Cache-aware routing that delegates to an external router.

    The external router is expected to expose:
    - route(request_body) -> int
    - prefill_complete(request_id)
    - generation_complete(request_id)

    The raw request body dict is passed directly to the external router,
    which is responsible for any tokenization or processing it needs.
    """

    def __init__(self, external_router: Any, clients: List[NeMoGymAsyncOpenAI]):
        self._external_router = external_router
        self._clients = clients

    def select_client(
        self,
        *,
        request_body: Dict[str, Any],
        request_id: str,
        session_id: Optional[str] = None,
    ) -> int:
        worker_idx = self._external_router.route(request_body)

        num_clients = len(self._clients)
        if not (0 <= worker_idx < num_clients):
            worker_idx = worker_idx % num_clients

        return worker_idx

    def on_prefill_complete(self, request_id: str) -> None:
        self._external_router.prefill_complete(request_id)

    def on_generation_complete(self, request_id: str) -> None:
        self._external_router.generation_complete(request_id)


# --- Config models ---


class RoundRobinRoutingPolicyConfig(BaseModel):
    type: Literal["round_robin"] = "round_robin"


class CacheAwareRoutingPolicyConfig(BaseModel):
    type: Literal["cache_aware"] = "cache_aware"
    router_class: str
    router_kwargs: Dict[str, Any] = Field(default_factory=dict)


RoutingPolicyConfig = Annotated[
    Union[RoundRobinRoutingPolicyConfig, CacheAwareRoutingPolicyConfig],
    Field(discriminator="type"),
]


def create_routing_policy(config: Union[RoundRobinRoutingPolicyConfig, CacheAwareRoutingPolicyConfig], clients: List[NeMoGymAsyncOpenAI]) -> RoutingPolicy:
    """Factory function to create a routing policy from a config."""
    if isinstance(config, RoundRobinRoutingPolicyConfig):
        return RoundRobinRoutingPolicy(clients)
    elif isinstance(config, CacheAwareRoutingPolicyConfig):
        module_path, class_name = config.router_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        router_cls = getattr(module, class_name)
        external_router = router_cls(**config.router_kwargs)
        return CacheAwareRoutingPolicy(external_router, clients)
    else:
        raise ValueError(f"Unknown routing policy config type: {type(config)}")
