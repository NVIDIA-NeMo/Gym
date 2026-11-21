# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact's task input: the ``task_input`` its episode request validates.

A row is a BIRD-Interact task as the benchmark publishes it, with the ground truth merged in; the Environment Server
keeps the ground truth from the agent. Data tooling loads this module without the server's dependencies, so it imports
only the standard library and pydantic.
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


TaskInput = BirdInteractTask
