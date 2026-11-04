# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto Ontology in the agent's process: its settings, its catalog and source databases, and its session calls.

Auto Ontology reads its settings (model endpoints, the catalog database, feature flags) from the process environment
when it is imported, and keeps its model clients for the whole process. So one Agent Server process holds one Auto
Ontology with one set of settings, and ``AutoOntologyRuntime`` imports it only after it has set them. Its calls are
synchronous: the agent runs them in threads.
"""

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class AskUser:
    """Auto Ontology asks the user a clarification question."""

    question: str


@dataclass(frozen=True)
class SubmitSQL:
    """Auto Ontology submits its SQL for the current phase."""

    sql: str


Action = AskUser | SubmitSQL
TurnKind = Literal["initial", "follow_up", "debug"]


class AutoOntologyRuntime:
    """The Auto Ontology of this process.

    ``environment`` holds the settings Auto Ontology reads at import. ``source_database_url`` is a connection string
    with a ``{database}`` field; Auto Ontology probes values in the database it names, and its connectors are kept for
    each database.
    """

    def __init__(self, environment: Mapping[str, str], source_database_url: str) -> None:
        self._environment = dict(environment)
        self._source_database_url = source_database_url
        self._lock = threading.Lock()
        self._modules: Optional[dict[str, Any]] = None
        self._connectors: dict[str, list[Any]] = {}

    def _auto_ontology(self) -> dict[str, Any]:
        with self._lock:
            if self._modules is None:
                os.environ.update(self._environment)
                from auto_ontology.connectors.registry import create_connector
                from auto_ontology.retrieval import interactive
                from auto_ontology.utils import get_data_objects_retriever, get_semantic_objects_retriever

                self._modules = {
                    "interactive": interactive,
                    "create_connector": create_connector,
                    "data_retriever": get_data_objects_retriever(),
                    "semantic_retriever": get_semantic_objects_retriever(),
                }
            return self._modules

    def _source_connectors(self, database: str) -> list[Any]:
        modules = self._auto_ontology()
        with self._lock:
            if database not in self._connectors:
                url = self._source_database_url.format(database=database)
                self._connectors[database] = [modules["create_connector"](url)]
            return self._connectors[database]

    def create_session(
        self,
        *,
        session_id: str,
        task_id: str,
        database: str,
        schema: str,
        knowledge: str,
        knowledge_children: dict[str, list[str]],
        question_budget: int,
        skip_evidence_refinement: bool,
        numeric_type_alignment: bool,
    ) -> Any:
        """Open an Auto Ontology session for one task. Its question arrives with the first ``step``."""
        modules = self._auto_ontology()
        return modules["interactive"].create_session(
            session_id=session_id,
            task_id=task_id,
            db_name=database,
            db_schema=schema,
            external_kb=knowledge,
            question="",
            data_retriever=modules["data_retriever"],
            semantic_retriever=modules["semantic_retriever"],
            connectors=self._source_connectors(database),
            max_clarify_turns=question_budget,
            external_kb_children_map=knowledge_children,
            skip_evidence_refinement=skip_evidence_refinement,
            numeric_type_alignment=numeric_type_alignment,
        )

    def step(self, session: Any, *, turn: TurnKind, text: Optional[str]) -> Action:
        """Advance ``session`` by one action.

        ``text`` is the question on an initial or follow-up turn, and the execution error on a debug turn (``None``
        when the SQL ran but its result was wrong).
        """
        interactive = self._auto_ontology()["interactive"]
        action = interactive.step(
            session,
            turn_type=interactive.TurnType(turn),
            initial_question=text if turn == "initial" else None,
            follow_up_question=text if turn == "follow_up" else None,
            debug_error=text if turn == "debug" else None,
        )
        if isinstance(action, interactive.AskUserAction):
            return AskUser(action.question)
        if isinstance(action, interactive.SubmitSQLAction):
            return SubmitSQL(action.sql)
        raise TypeError(f"Auto Ontology returned an unknown action: {type(action).__name__}")

    def apply_user_answer(self, session: Any, answer: str) -> None:
        self._auto_ontology()["interactive"].apply_user_answer(session, answer)

    def apply_submit_result(self, session: Any, result: dict) -> None:
        self._auto_ontology()["interactive"].apply_submit_result(session, result)
