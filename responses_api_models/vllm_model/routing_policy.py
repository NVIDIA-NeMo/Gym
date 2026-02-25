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
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

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

    def on_weights_updated(self) -> None:
        """Called when model weights have been updated. Default no-op."""


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


# --- Config models ---


class RoundRobinRoutingPolicyConfig(BaseModel):
    type: Literal["round_robin"] = "round_robin"


class CacheAwareRoutingPolicyConfig(BaseModel):
    """Config for an external cache-aware routing policy.

    ``router_class`` must be a fully-qualified class name that subclasses
    :class:`RoutingPolicy`. It will be instantiated with
    ``router_cls(clients=clients, **router_kwargs)``.
    """

    type: Literal["cache_aware"] = "cache_aware"
    router_class: str
    router_kwargs: Dict[str, Any] = Field(default_factory=dict)


RoutingPolicyConfig = Annotated[
    Union[RoundRobinRoutingPolicyConfig, CacheAwareRoutingPolicyConfig],
    Field(discriminator="type"),
]


def create_routing_policy(
    config: Union[RoundRobinRoutingPolicyConfig, CacheAwareRoutingPolicyConfig],
    clients: List[NeMoGymAsyncOpenAI],
) -> RoutingPolicy:
    """Factory function to create a routing policy from a config."""
    if isinstance(config, RoundRobinRoutingPolicyConfig):
        return RoundRobinRoutingPolicy(clients)
    elif isinstance(config, CacheAwareRoutingPolicyConfig):
        module_path, class_name = config.router_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        router_cls = getattr(module, class_name)
        if not (isinstance(router_cls, type) and issubclass(router_cls, RoutingPolicy)):
            raise TypeError(
                f"{config.router_class} must be a subclass of RoutingPolicy, got {router_cls}"
            )
        policy = router_cls(clients=clients, **config.router_kwargs)
        return policy
    else:
        raise ValueError(f"Unknown routing policy config type: {type(config)}")
