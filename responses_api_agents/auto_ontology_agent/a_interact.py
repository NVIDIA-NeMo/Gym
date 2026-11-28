# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BIRD-Interact's a-Interact for Auto Ontology's interactive pipeline.

The ``bird_interact`` Environment Server runs a-Interact as one activation (``environment_servers/bird_interact``):
its request names the database, the user's ambiguous request and the BIRD-COIN budget, and it grants nine tools, each
charged in BIRD-COIN. The agent runs the task as auto-ontology-eval runs it
(``ontology_sql_eval/bird_interact/server.py`` of https://github.com/NVIDIA/auto-ontology-eval at e727e4d6, Apache-2.0):
it buys the schema, the column meanings and the knowledge base first, then runs the pipeline with ``ask``, ``submit``
and ``execute`` for every query Auto Ontology runs on the database.

This module reads the request and names the tools; ``episode_runner.py`` runs the task.
"""

import json
import re
from dataclasses import dataclass
from typing import Any


ASK_TOOL = "ask"
SUBMIT_TOOL = "submit"
EXECUTE_TOOL = "execute"
SCHEMA_TOOL = "get_schema"
KNOWLEDGE_TOOL = "get_all_knowledge_definitions"
# What auto-ontology-eval buys before the task starts, in its order. It pays for the column meanings without reading
# them: the catalog holds them.
PURCHASES = (SCHEMA_TOOL, "get_all_column_meanings", KNOWLEDGE_TOOL)
# The tools the episode calls while it runs.
EPISODE_TOOLS = (ASK_TOOL, SUBMIT_TOOL, EXECUTE_TOOL)

_BUDGET = re.compile(r"your budget is (\d+(?:\.\d+)?)\.")
_REQUEST = re.compile(r"\ADatabase: ([A-Za-z0-9_-]+)\n\nMy request:\n(.*)\Z", re.DOTALL)


class AInteractTextError(ValueError):
    """A request that does not hold the bird_interact server's a-Interact texts."""


@dataclass(frozen=True)
class AInteractTask:
    database: str
    # The user's ambiguous request.
    question: str
    # The BIRD-COIN budget the task starts with.
    budget: float


def read_request(instructions: str, message: str) -> AInteractTask:
    """The task, from the a-Interact request's instructions and its message."""
    budget = _BUDGET.search(instructions)
    request = _REQUEST.match(message.strip())
    if budget is None or request is None:
        raise AInteractTextError("The request is not the bird_interact server's a-Interact request")
    return AInteractTask(database=request.group(1), question=request.group(2).strip(), budget=float(budget.group(1)))


def knowledge_entries(observation: str) -> list[dict[str, Any]]:
    """The visible knowledge-base entries in the reply of ``get_all_knowledge_definitions``."""
    try:
        entries = json.loads(observation)
    except json.JSONDecodeError as error:
        raise AInteractTextError(f"The knowledge base is not JSON: {observation[:200]!r}") from error
    if not isinstance(entries, list):
        raise AInteractTextError("The knowledge base is not a list")
    return entries
