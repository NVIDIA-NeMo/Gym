# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one episode of Auto Ontology's ReAct agent, inside the episode's sandbox.

    python episode_runner.py TASK_JSON RESULT_JSON

The Auto Ontology Agent Server delivers this file to the sandbox and runs it with the catalog image's Python, the one
that has Auto Ontology (https://github.com/NVIDIA/auto-ontology). It imports nothing from NeMo Gym.

TASK_JSON holds the task (``question``, ``evidence``, ``database``), the policy's Model Server URL and model name, the
image's settings file (``env_file``) and other settings (``environment``). The episode is the one Auto Ontology's own
``ReactSQLAgent`` runs for a SQL answer: ``run_episode`` with Auto Ontology's reasoning client, which the settings
point at the Model Server, so every model call goes through Gym on the rollout's route. RESULT_JSON gets the submitted
query, or none, and Auto Ontology's record of the episode.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any


# Auto Ontology's client needs a key; the Model Server holds the provider's.
MODEL_SERVER_API_KEY = "gym-model-server"


def episode_environment(task: dict[str, Any]) -> dict[str, str]:
    """The settings Auto Ontology reads at import: the image's, then the task's, then its model, which always wins."""
    settings: dict[str, str] = {}
    env_file = task.get("env_file")
    if env_file and Path(env_file).is_file():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    settings.update({str(key): str(value) for key, value in (task.get("environment") or {}).items()})
    settings.update(
        {
            "REASONING_ENDPOINT": task["model_base_url"],
            "REASONING_MODEL": task["model"],
            "REASONING_API_KEY": MODEL_SERVER_API_KEY,
        }
    )
    return settings


def run(task: dict[str, Any]) -> dict[str, Any]:
    """Answer the task's question with one ReAct episode, as Auto Ontology's ReactSQLAgent does."""
    os.environ.update(episode_environment(task))

    from auto_ontology.connectors import get_connectors
    from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
    from auto_ontology.retrieval.text_to_sql.react.loop import run_episode
    from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for
    from auto_ontology.utils import get_data_objects_retriever, get_semantic_objects_retriever
    from auto_ontology.utils.llm_invoke import get_llm_client

    database = task["database"]
    # The connectors the catalog was ingested with (CONNECTION_STRINGS), so their names match the catalog's.
    connectors = {connector.database_name: connector for connector in get_connectors()}
    if database not in connectors:
        raise KeyError(f"No source database {database!r}; the catalog has {sorted(connectors)}")
    data_retriever = get_data_objects_retriever()
    context = ToolContext(
        retriever=get_semantic_objects_retriever() or data_retriever,
        data_retriever=data_retriever,
        connectors=[connectors[database]],
        database_name=database,
        information_answers=False,
        question=task["question"],
    )
    result = run_episode(
        get_llm_client(),
        context,
        question=task["question"],
        evidence=task.get("evidence") or None,
        dialect=getattr(connector_for(context), "dialect", None),
    )
    submission = result.submission
    ran = submission is not None and submission.db_response is not None
    return {
        "sql": submission.sql if ran else None,
        "reason": result.reason,
        "tool_calls": result.tool_calls,
        "trajectory": result.trajectory,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    task_path, result_path = (Path(arg) for arg in argv)
    result = run(json.loads(task_path.read_text()))
    result_path.write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
