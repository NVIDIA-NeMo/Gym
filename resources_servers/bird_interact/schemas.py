# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route contracts of the BIRD-Interact Resources Server. The task rows are in ``task_data.py``."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse, ResourcesSeedSessionResponse


InteractionMode = Literal["c_interact", "a_interact"]


class EpisodeContext(BaseModel):
    """What the agent sees at the start of an episode.

    In a-Interact the agent pays BIRD-COIN for the schema, the column meanings and the knowledge base, so the context
    leaves them out.
    """

    model_config = ConfigDict(extra="forbid")

    database: str
    question: str
    # c-Interact only.
    schema_text: str | None = None
    column_meanings: str | None = None
    knowledge: str | None = None
    # c-Interact: clarifying questions the agent may ask. a-Interact: None.
    question_budget: int | None = None
    # a-Interact: BIRD-COIN for all actions. c-Interact: None.
    coin_budget: float | None = None


class BirdInteractSeedSessionResponse(ResourcesSeedSessionResponse):
    """The seed reply, with the context for the Environment Server's ``responses_create_params``."""

    context: EpisodeContext


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="The question.")


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    # False when the question was not allowed, such as after the question budget ran out.
    answered: bool = True
    questions_left: int | None = None


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str


class SubmitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    # What the user says next: the debug prompt, the follow-up question, or the end of the task.
    feedback: str
    finished: bool
    phase: int


class SQLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(description="The SQLite query.")


class ColumnMeaningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(description="The table.")
    column_name: str = Field(description="The column.")


class KnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_name: str = Field(description="The entry's name.")


# The a-Interact tools as the agent sees them over either transport: each tool's description, and the request model of
# the Resources Server route that serves it (None for a tool without arguments).
A_INTERACT_TOOLS: dict[str, tuple[str, type[BaseModel] | None]] = {
    "ask": ("Ask the user one clarifying question.", AskRequest),
    "submit": ("Submit your final SQL for the current question.", SQLRequest),
    "execute": ("Run an exploratory SQLite query and see up to 20 rows.", SQLRequest),
    "get_schema": ("Read the database schema, with sample rows.", None),
    "get_all_column_meanings": ("Read the meaning of every column.", None),
    "get_column_meaning": ("Read the meaning of one column.", ColumnMeaningRequest),
    "get_all_external_knowledge_names": ("List the names of the knowledge-base entries.", None),
    "get_knowledge_definition": ("Read one knowledge-base entry.", KnowledgeRequest),
    "get_all_knowledge_definitions": ("Read every knowledge-base entry.", None),
}


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


class BirdInteractVerifyRequest(BaseVerifyRequest):
    pass


class BirdInteractVerifyResponse(BaseVerifyResponse):
    phases: list[PhaseResult]
    questions_asked: int
    coins_spent: float
    finished: bool
