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
Shared schemas for the Computer Use resource server and agent.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import NeMoGymResponse


########################################
# Resource Server Config
########################################


class BrowserGymResourcesServerConfig(BaseResourcesServerConfig):
    max_concurrent_browsers: int = 16
    browser_pool_size: int = 4
    default_viewport_width: int = 1280
    default_viewport_height: int = 720
    verify_connector_limit: int = 512
    verify_connector_limit_per_host: int = 128
    verify_timeout_seconds: float = 300.0
    session_ttl_seconds: float = 7200.0
    session_reaper_interval_seconds: float = 300.0


########################################
# Browser Action (superset of all providers)
########################################


class BrowserAction(BaseModel):
    """Unified browser action schema -- superset of OpenAI, Anthropic, and Gemini action types."""

    action_type: str
    coordinate: Optional[List[int]] = None
    text: Optional[str] = None
    scroll_x: Optional[int] = None
    scroll_y: Optional[int] = None
    scroll_direction: Optional[str] = None
    scroll_amount: Optional[int] = None
    start_coordinate: Optional[List[int]] = None
    end_coordinate: Optional[List[int]] = None
    path: Optional[List[List[int]]] = None
    key: Optional[str] = None
    keys: Optional[List[str]] = None
    button: Optional[str] = None
    duration: Optional[int] = None
    url: Optional[str] = None
    tab_index: Optional[int] = None
    press_enter: Optional[bool] = None
    clear_before_typing: Optional[bool] = None
    region: Optional[List[int]] = None


########################################
# Seed Session
########################################


class CUASeedSessionRequest(BaseSeedSessionRequest):
    start_url: str
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None


class CUASeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    screenshot: str  # base64-encoded PNG


########################################
# Step
########################################


class CUAStepRequest(BaseModel):
    env_id: str
    action: BrowserAction


class CUAStepResponse(BaseModel):
    screenshot: str  # base64-encoded PNG
    current_url: str


########################################
# Dump Local Storage
########################################


class CUADumpLocalStorageRequest(BaseModel):
    env_id: str


class CUADumpLocalStorageResponse(BaseModel):
    local_storage_dump: str  # JSON string of localStorage contents


########################################
# Close
########################################


class CUACloseRequest(BaseModel):
    env_id: str


class CUACloseResponse(BaseModel):
    message: str
    success: bool


########################################
# Trajectory (for RL training data)
########################################


class CUAStep(BaseModel):
    """A single step in the CUA trajectory."""

    action: BrowserAction
    screenshot_before: Optional[str] = None  # base64
    screenshot_after: str  # base64
    current_url: str
    raw_provider_response: Dict[str, Any] = Field(default_factory=dict)
    prompt_token_ids: List[int] = Field(default_factory=list)
    generation_token_ids: List[int] = Field(default_factory=list)
    generation_log_probs: List[float] = Field(default_factory=list)


class CUATrajectory(BaseModel):
    """Full trajectory of a CUA rollout for RL training."""

    steps: List[CUAStep] = Field(default_factory=list)
    task_prompt: str = ""
    initial_screenshot: str = ""  # base64
    final_message: Optional[str] = None


########################################
# NeMoGymResponse extension for CUA
########################################


class CUANeMoGymResponse(NeMoGymResponse):
    model_config = ConfigDict(extra="allow")

    env_id: str
    trajectory: CUATrajectory
    local_storage_dump: Optional[str] = None


########################################
# Verify Request / Response (MRO pattern from aviary)
########################################


class CUAVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: CUANeMoGymResponse
    verifier_metadata: Optional[Dict[str, Any]] = None


class CUAVerifyResponse(CUAVerifyRequest, BaseVerifyResponse):
    """MRO: CUAVerifyRequest.response (CUANeMoGymResponse) supersedes BaseVerifyResponse.response."""

    model_config = ConfigDict(extra="allow")
