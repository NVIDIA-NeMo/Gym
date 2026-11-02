# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The a-Interact tools' arguments and replies, and the per-phase results. The task rows are in ``task_data.py``."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


InteractionMode = Literal["c_interact", "a_interact"]


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="The question.")


class SQLArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(description="The SQLite query.")


class ColumnMeaningArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(description="The table.")
    column_name: str = Field(description="The column.")


class KnowledgeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_name: str = Field(description="The entry's name.")


class ToolObservation(BaseModel):
    """What an a-Interact tool returns to the agent."""

    model_config = ConfigDict(extra="forbid")

    observation: str
    coins_left: float


class PhaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: int
    passed: bool
    attempts: int
    reward: float
