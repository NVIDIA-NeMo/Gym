# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in configuration for partial-rollout checkpointing."""

from collections.abc import Mapping
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


CHECKPOINT_BLOCK = "checkpoint"


class CheckpointSettings(BaseModel):
    """Global ``checkpoint:`` block.

    Checkpointing is off unless ``enabled`` is true.
    When it is off, no participant is installed and servers keep no checkpoint bookkeeping.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    control_auth_token: Optional[str] = Field(
        default=None,
        description="Bearer token required on every /ng-control/v1/checkpoint route.",
    )
    lease_grace_seconds: float = Field(
        default=600.0,
        gt=0,
        description="How long past a control call's deadline a participant stays closed without hearing from "
        "the controller before it resumes on its own.",
    )

    @model_validator(mode="after")
    def require_token_when_enabled(self) -> Self:
        if self.enabled and not self.control_auth_token:
            raise ValueError("checkpoint.control_auth_token is required when checkpoint.enabled is true")
        return self


def checkpoint_settings(global_config: Any) -> Optional[CheckpointSettings]:
    """Return enabled checkpoint settings, or ``None`` when checkpointing is off."""
    if not isinstance(global_config, (Mapping, DictConfig)):
        return None
    block = global_config.get(CHECKPOINT_BLOCK)
    if block is None:
        return None
    if isinstance(block, DictConfig):
        block = OmegaConf.to_container(block, resolve=True)
    settings = CheckpointSettings.model_validate(block)
    return settings if settings.enabled else None
