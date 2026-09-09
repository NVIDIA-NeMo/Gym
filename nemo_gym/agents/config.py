# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class AgentModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    provider: str = ""
    api_key: str = ""  # pragma: allowlist secret
    base_url: Optional[str] = None
    settings: dict[str, Any] = Field(default_factory=dict)


class AgentHarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: AgentModelConfig
    timeout_seconds: float = Field(default=300, gt=0)
    max_turns: Optional[int] = Field(default=None, gt=0)
    system_prompt: Optional[str] = None
    workspace: Optional[Path] = None
    settings: dict[str, Any] = Field(default_factory=dict)
