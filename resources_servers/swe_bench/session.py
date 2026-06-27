# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SessionDescriptor — Environment response after accepting a Task.

The descriptor is **episode context**, not the Task itself: placement topology,
sandbox spec, egress hints, and a round-trip verifier payload for ``/verify``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from resources_servers.swe_bench.task import ENVIRONMENT_NAME, TaskPublic


Topology = Literal["none", "env_sandboxed", "agent_in_env", "whole_interaction"]


class PlacementDescriptor(BaseModel):
    topology: Topology


class SandboxDescriptor(BaseModel):
    spec: dict[str, Any]


class EgressDescriptor(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class SessionDescriptor(BaseSeedSessionResponse):
    """Environment-owned episode context returned from ``seed_session``."""

    environment: str = ENVIRONMENT_NAME
    task: TaskPublic
    placement: PlacementDescriptor
    sandbox: SandboxDescriptor
    egress: EgressDescriptor
    verifier_metadata: dict[str, Any]


class SweBenchSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[dict[str, Any]] = None


class SweBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[dict[str, Any]] = None


class SweBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str = ""
    environment: str = ENVIRONMENT_NAME
    resolved: bool = False
    patch_exists: bool = False
    mask_sample: bool = False
    error_kind: Optional[str] = None


SweBenchSeedSessionResponse = SessionDescriptor
