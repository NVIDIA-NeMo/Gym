# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-data schema for BIRD-Interact rows.

A row is a BIRD-Interact task as the benchmark publishes it, with the ground truth merged in. It is the episode's
task input; the Environment Server keeps the ground truth from the agent.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


_PROVENANCE = {"consumed_by": ["provenance"]}
# A phase of this category is graded by comparing its result with the solution's, whatever its test cases.
QUERY_CATEGORY = "Query"


class AmbiguityPoint(BaseModel):
    """An annotated ambiguity: the term in the question and the SQL fragment that resolves it.

    The fields are declared in the published rows' order, so ``model_dump(exclude_unset=True)`` writes a row's
    ambiguities as the simulator's prompt shows them.
    """

    model_config = ConfigDict(extra="allow")

    term: str
    sql_snippet: str
    is_mask: bool | None = None
    type: str | None = None


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
    test_cases: list[str] = Field(default_factory=list)
    category: str = QUERY_CATEGORY


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
    # Query or Management. A Management phase runs its custom test cases, on PostgreSQL.
    category: str = QUERY_CATEGORY
    # The question without its ambiguities. The simulator reads it when the row has it.
    query: str | None = None
    follow_up: FollowUp | None = None
    # Fields of the published rows that Gym does not read.
    high_level: bool | None = Field(default=None, json_schema_extra=_PROVENANCE)
    external_knowledge: list[Any] = Field(default_factory=list, json_schema_extra=_PROVENANCE)
    # The reference harness resets its shared database with these. Each Gym session has its own sandbox.
    clean_up_sqls: list[str] = Field(default_factory=list, json_schema_extra=_PROVENANCE)

    @property
    def ambiguity_count(self) -> int:
        """Critical and knowledge ambiguities: the questions the agent needs to resolve the task."""
        return len(self.user_query_ambiguity.critical_ambiguity) + len(self.knowledge_ambiguity)

    def needs_custom_test_cases(self) -> bool:
        """Whether a phase is graded by its custom test cases (see ``custom_test_cases``)."""
        phases = [self] if self.follow_up is None else [self, self.follow_up]
        return any(custom_test_cases(phase) for phase in phases)


def custom_test_cases(phase: BirdInteractTask | FollowUp) -> list[str]:
    """The test cases that grade a phase, as BIRD-Interact grades it: a phase that is not a query runs its own. A
    query, or a phase with none, is graded by comparing its result with the solution's, so it gets an empty list."""
    return [] if phase.category == QUERY_CATEGORY else phase.test_cases


TaskData = BirdInteractTask
