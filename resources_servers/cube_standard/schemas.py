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
"""
Pydantic request/response schemas for the CubeResourcesServer endpoints.

These schemas are shared with the CubeAgent so that both sides agree on
the wire format for /seed_session, /step, /verify, and /close.
"""

from typing import Any, Dict, List, Optional

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import NeMoGymEasyInputMessage


class CubeSeedSessionRequest(BaseSeedSessionRequest):
    """Request body for POST /seed_session."""

    task_id: Optional[str] = Field(
        default=None,
        description="Specific task ID to run. If None, one is selected (round-robin or random).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed to apply to the selected task config.",
    )


class CubeSeedSessionResponse(BaseSeedSessionResponse):
    """Response body for POST /seed_session."""

    obs: List[NeMoGymEasyInputMessage] = Field(
        description="Initial observation as NeMo Gym messages (from task.reset())."
    )
    tools: List[FunctionToolParam] = Field(
        description="Tool definitions in FunctionToolParam format (from task.action_set)."
    )
    task_id: str = Field(description="The task ID that was selected.")


class CubeStepRequest(BaseModel):
    """Request body for POST /step."""

    call_id: str = Field(
        description="Tool call ID from the model (NeMoGymResponseFunctionToolCall.call_id)."
    )
    name: str = Field(description="Tool/action name.")
    arguments: Dict[str, Any] = Field(
        description="Parsed arguments dict (NOT JSON string — agent calls json.loads before sending)."
    )


class CubeStepResponse(BaseModel):
    """Response body for POST /step."""

    output: str = Field(
        description=(
            "For text observations: the observation text. "
            "For image observations: a URL to the screenshot file on /screenshots/."
        )
    )
    content_type: str = Field(
        default="text/plain",
        description=(
            "MIME type of output. 'text/plain' for text, 'image/png' for screenshots. "
            "CubeAgent uses this to decide how to inject the observation into the conversation."
        ),
    )
    done: bool = Field(description="True when the episode has ended.")
    reward: float = Field(
        default=0.0,
        description="Reward from this step (non-zero only when done=True by default).",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error string if StepError occurred.",
    )


class CubeVerifyRequest(BaseVerifyRequest):
    """Request body for POST /verify. Session identified by cookie."""

    pass


class CubeVerifyResponse(BaseVerifyResponse):
    """Response body for POST /verify."""

    reward_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra info from task.evaluate() / EnvironmentOutput.info.",
    )


class CubeCloseRequest(BaseModel):
    """Request body for POST /close. Session identified by cookie."""

    pass


class CubeCloseResponse(BaseModel):
    """Response body for POST /close."""

    message: str
    success: bool
