# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the tales (TALES text-adventure suite) server.

Gymnasium-family environment: no /verify — scoring happens in /reset + /step, driven by
gymnasium_agent. The row is a POINTER into the external ``tales`` package's environment
registry: reset() imports ``tales.<framework>`` and indexes its ``environments`` /
``train_environments`` list with ``task_no``; the task content itself lives in that package, not
in the row. All fields arrive untyped via ``EnvResetRequest.model_extra``
(``resources_servers/gymnasium/base.py:33``, ``extra="allow"``) and are resolved defensively —
``metadata.get(key)`` falling back to ``TALESResourcesServerConfig`` defaults (``app.py:136``) —
so every field is Optional on the wire.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    framework: Optional[str] = Field(
        default=None,
        description=(
            "TALES framework module to load (tales.<framework>), e.g. textworld, textworld_express, "
            'alfworld, scienceworld, jericho. Config default "textworld" when absent. Deliberately '
            "not an enum: the set is owned by the external tales package."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_no: Optional[int] = Field(
        default=None,
        description=(
            "Index into the framework's environment registry (train_environments for split=train, "
            "else environments). Config default 0 when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    split: Optional[str] = Field(
        default=None,
        description=(
            'Registry split ("train" selects train_environments when the framework defines them); '
            'also forwarded to gym.make for scienceworld. Config default "train" when absent.'
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed passed to env.reset(). Config default 0 when absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    max_episode_steps: Optional[int] = Field(
        default=None,
        description=(
            "Per-episode step budget before truncation. Accepted per-row; absent in committed data "
            "(config default 25)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
