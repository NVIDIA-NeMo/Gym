# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Required native runtime settings, interpreted only by the selected adapter."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class AgentRuntimePolicy(BaseModel):
    """Preserve a versioned policy without orchestration translating its semantics.

    Adapters must reject unsupported formats or settings before runtime setup.
    Policy settings are part of the immutable, idempotent session seed request.
    """

    model_config = ConfigDict(extra="forbid")

    format: str = Field(min_length=1)
    settings: dict[str, JsonValue]
