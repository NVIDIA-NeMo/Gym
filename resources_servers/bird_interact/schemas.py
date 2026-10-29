# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route contracts of the BIRD-Interact Resources Server. The task rows are in ``task_data.py``."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

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

    question: str


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

    sql: str


class ColumnMeaningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: str
    column_name: str


class KnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_name: str


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
