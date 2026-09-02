# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the example_mcp_weather server.

The one example server whose rows use the legacy ``verifier_metadata`` bucket. Schemas are
written flat (the end-state shape): core validation splices ``verifier_metadata`` contents up
before checking, and the ``legacy_location`` annotation records where the field lives on today's
wire so migration tooling can map both directions.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_city: str = Field(
        default="Paris",
        description="City the agent must fetch weather for; verify() defaults to Paris if absent.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
