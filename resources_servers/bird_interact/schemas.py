# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task rows and route contracts of the BIRD-Interact Resources Server."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse, ResourcesSeedSessionResponse


InteractionMode = Literal["c_interact", "a_interact"]


class AmbiguityPoint(BaseModel):
    """An annotated ambiguity: the term in the question and the SQL fragment that resolves it."""

    model_config = ConfigDict(extra="allow")

    term: str
    sql_snippet: str


class KnowledgeAmbiguity(AmbiguityPoint):
    """An ambiguity from a knowledge-base entry that is hidden from the agent."""

    deleted_knowledge: int


class UserQueryAmbiguity(BaseModel):
    model_config = ConfigDict(extra="allow")

    critical_ambiguity: list[AmbiguityPoint] = Field(default_factory=list)
    non_critical_ambiguity: list[AmbiguityPoint] = Field(default_factory=list)


class Conditions(BaseModel):
    model_config = ConfigDict(extra="allow")

    order: bool = False


class FollowUp(BaseModel):
    """The second question, asked after the first is solved."""

    model_config = ConfigDict(extra="allow")

    query: str
    sol_sql: list[str] = Field(min_length=1)
    conditions: Conditions = Field(default_factory=Conditions)


class BirdInteractTask(BaseModel):
    """One BIRD-Interact row. ``sol_sql`` is ground truth, which the public releases leave empty."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    selected_database: str
    amb_user_query: str
    user_query_ambiguity: UserQueryAmbiguity
    knowledge_ambiguity: list[KnowledgeAmbiguity] = Field(default_factory=list)
    sol_sql: list[str]
    preprocess_sql: list[str] = Field(default_factory=list)
    test_cases: list[str] = Field(default_factory=list)
    conditions: Conditions = Field(default_factory=Conditions)
    # The question without its ambiguities. The simulator reads it when the row has it.
    query: str | None = None
    follow_up: FollowUp | None = None

    @property
    def ambiguity_count(self) -> int:
        """Critical and knowledge ambiguities: the questions the agent needs to resolve the task."""
        return len(self.user_query_ambiguity.critical_ambiguity) + len(self.knowledge_ambiguity)

    def ambiguities(self) -> list[AmbiguityPoint]:
        query = self.user_query_ambiguity
        return [*query.critical_ambiguity, *query.non_critical_ambiguity, *self.knowledge_ambiguity]


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
    """The seed reply, with the context that the Environment Server's opening request shows."""

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
