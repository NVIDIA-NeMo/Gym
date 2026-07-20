"""HTTP models for Gym's stateful OSWorld Resources Server."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)


class EnvironmentOptions(BaseModel):
    action_space: str = "pyautogui"
    screen_width: int = Field(default=1920, ge=1)
    screen_height: int = Field(default=1080, ge=1)
    headless: bool = True
    require_a11y_tree: bool = False
    require_terminal: bool = False
    client_password: str = "password"  # pragma: allowlist secret
    enable_proxy: bool = False


class OSWorldSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    task_config: Dict[str, Any]
    environment: EnvironmentOptions = Field(default_factory=EnvironmentOptions)


class OSWorldResetRequest(BaseModel):
    task_config: Dict[str, Any]


class OSWorldStepRequest(BaseModel):
    operation_id: str
    action: Any
    pause: float = Field(default=0.5, ge=0.0, le=120.0)


class OSWorldObservation(BaseModel):
    screenshot_b64: str = ""
    accessibility_tree: Any = None
    terminal: Any = None
    instruction: Optional[str] = None

    @classmethod
    def from_observation(cls, observation: Dict[str, Any]) -> "OSWorldObservation":
        screenshot = observation.get("screenshot") or b""
        if isinstance(screenshot, str):
            screenshot = screenshot.encode("utf-8")
        return cls(
            screenshot_b64=base64.b64encode(screenshot).decode("ascii"),
            accessibility_tree=observation.get("accessibility_tree"),
            terminal=observation.get("terminal"),
            instruction=observation.get("instruction"),
        )


class OSWorldSeedSessionResponse(BaseSeedSessionResponse):
    session_id: str
    task_id: str
    worker: str
    status: str
    observation: OSWorldObservation


class OSWorldStepResponse(BaseModel):
    operation_id: str
    observation: OSWorldObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class OSWorldEvaluateResponse(BaseModel):
    score: float


class OSWorldCloseResponse(BaseModel):
    closed: bool


class OSWorldSessionStatusResponse(BaseModel):
    session_id: str
    task_id: str
    worker: str
    status: str
    created_at: float
    last_access_at: float


class OSWorldVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class OSWorldVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    osworld_score: float
    mask_sample: bool = False
