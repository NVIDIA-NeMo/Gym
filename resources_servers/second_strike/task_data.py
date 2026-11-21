# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the second_strike server.

Gymnasium-family environment (see ``resources_servers/gymnasium/base.py``): there is no /verify; the reward (the
balanced Midnight Score) comes from the last /step, driven by gymnasium_agent. Each row names one fixed war of the
Midnight Index. reset() reads ``scenario`` and ``seed`` from the row extras (``EnvResetRequest.model_extra``) and
joins that war on the SECOND STRIKE game server, so both are required: a row without them raises at reset().
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenario: Literal["ffa", "cornered", "hegemon", "laststand", "treaty", "surrender"] = Field(
        description="The Midnight Index scenario, sent to the game server as the war's index.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    seed: int = Field(
        description="The war's seed. The same scenario and seed always give the same states in lockstep.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
