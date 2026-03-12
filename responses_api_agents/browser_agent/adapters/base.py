# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
Base class for CUA adapters. Each adapter manages provider-specific API calls
and context management internally, exposing a simple initialize/step/reset interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from resources_servers.browser_gym.schemas import BrowserAction


class CUAAdapterResponse(BaseModel):
    """Response from an adapter step."""

    actions: List[BrowserAction] = Field(default_factory=list)
    message: Optional[str] = None
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False


class BaseCUAAdapter(ABC):
    """Abstract base class for CUA provider adapters.

    Each adapter internally manages its own context/history and exposes
    only the simple initialize()/step()/reset() interface to the agent.
    """

    @abstractmethod
    async def initialize(self, task_prompt: str, screenshot_b64: str) -> CUAAdapterResponse:
        """First call with initial prompt + screenshot. Sets up provider context."""
        ...

    @abstractmethod
    async def step(self, screenshot_b64: str, action_result: Optional[str] = None) -> CUAAdapterResponse:
        """Follow-up call with new screenshot after action execution. Manages context internally."""
        ...

    @abstractmethod
    def reset(self):
        """Clear adapter state (context, history) for reuse with a new task."""
        ...
