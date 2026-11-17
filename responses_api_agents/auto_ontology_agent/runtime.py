# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto Ontology's ReAct agent in the agent's process: its settings, its catalog and source databases, its episodes.

Auto Ontology reads its settings (the model endpoint, the catalog database, the embedding endpoint and the ReAct
agent's limits and tools) from the process environment when it is imported. So one Agent Server process holds one
Auto Ontology with one set of settings, and ``AutoOntologyRuntime`` imports it only after it has set them.
``run_episode`` is synchronous: the agent runs it in a thread. Its chat model sends through the agent's HTTP client
(model_calls.py).
"""

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass(frozen=True)
class Episode:
    """How one ReAct episode ended."""

    # The query the model submitted, once it passed Auto Ontology's checks and ran; None without one.
    sql: Optional[str]
    # Auto Ontology's reason: submitted, no_submission, budget_exhausted, model_error, ...
    reason: str
    tool_calls: int
    # Auto Ontology's own record of the episode: each message, with tool results clipped.
    trajectory: list[dict[str, Any]] = field(default_factory=list)


class AutoOntologyRuntime:
    """The Auto Ontology of this process.

    ``environment`` holds the settings Auto Ontology reads at import. ``source_database_url`` is a connection string
    with a ``{database}`` field; ``run_sql`` runs on the database it names, and its connectors are kept for each
    database. The chat model's calls go through ``http_client``.
    """

    def __init__(
        self,
        environment: Mapping[str, str],
        source_database_url: str,
        *,
        http_client: httpx.Client,
        numeric_type_alignment: bool = False,
    ) -> None:
        self._environment = dict(environment)
        self._source_database_url = source_database_url
        self._http_client = http_client
        self._numeric_type_alignment = numeric_type_alignment
        self._lock = threading.Lock()
        self._modules: Optional[dict[str, Any]] = None
        self._connectors: dict[str, list[Any]] = {}

    def _auto_ontology(self) -> dict[str, Any]:
        with self._lock:
            if self._modules is None:
                os.environ.update(self._environment)
                from auto_ontology.connectors.registry import create_connector
                from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
                from auto_ontology.retrieval.text_to_sql.react.loop import run_episode
                from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for
                from auto_ontology.utils import get_data_objects_retriever, get_semantic_objects_retriever
                from auto_ontology.utils.llm_invoke import get_llm_client

                self._modules = {
                    "create_connector": create_connector,
                    "ToolContext": ToolContext,
                    "run_episode": run_episode,
                    "connector_for": connector_for,
                    # The reasoning client, which Auto Ontology's own ReAct step gives the episode.
                    "llm": send_through(get_llm_client(), self._http_client),
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

    def run_episode(self, *, question: str, evidence: str, database: str) -> Episode:
        """Answer ``question`` on ``database`` with one ReAct episode, as Auto Ontology's ``ReactSQLAgent`` does.

        As auto-ontology-eval runs it for a graded benchmark: SQL answers only (``calculation_only``, so no
        information answers), with the question as asked and the evidence apart.
        """
        modules = self._auto_ontology()
        retriever, data_retriever = modules["semantic_retriever"], modules["data_retriever"]
        ctx = modules["ToolContext"](
            retriever=retriever or data_retriever,
            data_retriever=data_retriever,
            connectors=list(self._source_connectors(database)),
            database_name=database,
            numeric_type_alignment=self._numeric_type_alignment,
            information_answers=False,
            question=question,
        )
        connector = modules["connector_for"](ctx)
        result = modules["run_episode"](
            modules["llm"],
            ctx,
            question=question,
            evidence=evidence or None,
            dialect=getattr(connector, "dialect", None),
        )
        submission = result.submission
        sql = submission.sql if submission is not None and submission.db_response is not None else None
        return Episode(sql=sql, reason=result.reason, tool_calls=result.tool_calls, trajectory=list(result.trajectory))


def send_through(llm: Any, http_client: httpx.Client) -> Any:
    """Give Auto Ontology's chat client ``http_client``, keeping the base URL, timeout and retries it was built with.

    Auto Ontology's ``get_llm_client`` takes no HTTP client; only its ChatOpenAI client, which a model name with an
    ``openai/``, ``azure/`` or ``aws/`` prefix selects, can take one.
    """
    from langchain_openai import ChatOpenAI

    if not isinstance(llm, ChatOpenAI):
        raise TypeError(f"Auto Ontology built a {type(llm).__name__}; only its ChatOpenAI client can send through Gym")
    llm.root_client = llm.root_client.with_options(http_client=http_client)
    llm.client = llm.root_client.chat.completions
    return llm
