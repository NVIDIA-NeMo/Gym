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


class CUAAdapterUsage(BaseModel):
    """Provider-agnostic token usage from an adapter call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class CUAAdapterResponse(BaseModel):
    """Response from an adapter step."""

    actions: List[BrowserAction] = Field(default_factory=list)
    message: Optional[str] = None
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    usage: Optional[CUAAdapterUsage] = None
    prompt_token_ids: List[int] = Field(default_factory=list)
    generation_token_ids: List[int] = Field(default_factory=list)
    generation_log_probs: List[float] = Field(default_factory=list)


def extract_token_ids_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract RL-specific token ID fields from a model server response.

    Checks multiple locations where token IDs may appear:
    1. On the last output item (vLLM /v1/responses format)
    2. In provider_specific_fields
    3. At the top level of the response

    Returns a dict with prompt_token_ids, generation_token_ids, generation_log_probs
    (all defaulting to empty lists if not found).
    """
    result: Dict[str, Any] = {
        "prompt_token_ids": [],
        "generation_token_ids": [],
        "generation_log_probs": [],
    }
    keys = list(result.keys())

    for item in reversed(response.get("output", [])):
        if isinstance(item, dict) and any(k in item for k in keys):
            for k in keys:
                result[k] = item.get(k, [])
            return result

    psf = response.get("provider_specific_fields", {})
    if isinstance(psf, dict) and any(k in psf for k in keys):
        for k in keys:
            result[k] = psf.get(k, [])
        return result

    if any(k in response for k in keys):
        for k in keys:
            result[k] = response.get(k, [])

    return result


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
