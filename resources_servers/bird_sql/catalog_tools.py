# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The catalog tools of the ``bird_sql`` catalog Resources Server: their arguments and their definitions.

The arguments are those of Auto Ontology's tool functions (https://github.com/NVIDIA/auto-ontology,
``auto_ontology/retrieval/text_to_sql/react``), which the tools call in the episode's sandbox. The descriptions are
Gym's. ``tool_definitions()`` gives the tools as Responses API function tools, for the rows that offer them; the
routes in ``catalog_app.py`` take the same models as their bodies.

This module imports only Pydantic, so the benchmark's prepare script can use it.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# What a submission the catalog accepted returns. verify grades the last submission that returned it.
ACCEPTED = "Accepted."


class RetrieveArguments(BaseModel):
    """Search the database's catalog: its tables, columns, business terms, definitions and saved analyses.

    Use it before writing SQL, to find the tables and columns a question needs and what their values mean.
    """

    model_config = ConfigDict(extra="forbid", title="retrieve")

    query: str = Field(description="What to look for, in words, such as 'free meal count by county'.")
    kinds: Optional[list[Literal["term", "attribute", "definition", "analysis"]]] = Field(
        default=None,
        description=(
            "Limit the search to these kinds: 'term' (business concepts and their tables), 'attribute' (columns), "
            "'definition' (named calculations) or 'analysis' (saved example queries). All kinds when omitted."
        ),
    )


class RunSqlArguments(BaseModel):
    """Run a read-only SQL query on the database and see its first rows.

    Use it to check values and to test a draft query. The query is checked against the catalog first.
    """

    model_config = ConfigDict(extra="forbid", title="run_sql")

    sql: str = Field(description="One SQLite query.")


class SubmitArguments(BaseModel):
    """Submit the final SQL query that answers the question. It is checked and run.

    When it is accepted, the episode ends. When it is not, the reply says why, and you can fix it and submit again.
    """

    model_config = ConfigDict(extra="forbid", title="submit")

    sql: str = Field(description="The final SQLite query. Return exactly the columns the question asks for.")
    response: str = Field(default="", description="A short answer to the question, in words.")
    assumptions: str = Field(default="", description="How you read anything the question left open.")


TOOL_ARGUMENTS: dict[str, type[BaseModel]] = {
    "retrieve": RetrieveArguments,
    "run_sql": RunSqlArguments,
    "submit": SubmitArguments,
}


def tool_definitions() -> list[dict[str, Any]]:
    """The catalog tools as Responses API function tools."""
    definitions = []
    for name, arguments in TOOL_ARGUMENTS.items():
        schema = arguments.model_json_schema()
        description = " ".join((arguments.__doc__ or "").split())
        schema.pop("description", None)
        schema.pop("title", None)
        for field in schema.get("properties", {}).values():
            field.pop("title", None)
        definitions.append(
            {"type": "function", "name": name, "description": description, "parameters": schema, "strict": False}
        )
    return definitions
