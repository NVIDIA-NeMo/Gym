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
query, or none, Auto Ontology's record of the episode, and the model call that ended it when the loop gave up on one
(``model_failure``).

An Environment Server's tools can take part, for a benchmark such as BIRD-Interact whose user the agent asks.
``tools`` are offered to the model beside Auto Ontology's own, each called over HTTP at its ``url``; ``submit`` is where
the query Auto Ontology accepts goes, as ``{"sql": ...}``. ``custom_prompts`` extends the system prompt. RESULT_JSON
then also gets each call of those tools and the reply to the submission.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from string import Template
from typing import Any


# Auto Ontology's client needs a key; the Model Server holds the provider's.
MODEL_SERVER_API_KEY = "gym-model-server"
# A call of an Environment Server's tool, such as a question its simulated user answers.
TOOL_TIMEOUT_SECONDS = 600


def episode_environment(task: dict[str, Any]) -> dict[str, str]:
    """The settings Auto Ontology reads at import: the image's, then the task's, then its model, which always wins.

    A value may name another setting as ``${NAME}``, such as a source database on the catalog's ``${POSTGRES_PORT}``,
    which each episode sets.
    """
    settings: dict[str, str] = {}
    env_file = task.get("env_file")
    if env_file and Path(env_file).is_file():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    settings.update({str(key): str(value) for key, value in (task.get("environment") or {}).items()})
    settings = {key: Template(value).safe_substitute(settings) for key, value in settings.items()}
    settings.update(
        {
            "REASONING_ENDPOINT": task["model_base_url"],
            "REASONING_MODEL": task["model"],
            "REASONING_API_KEY": MODEL_SERVER_API_KEY,
        }
    )
    return settings


def model_failure(error: BaseException) -> dict[str, Any]:
    """What Gym's outcome policy reads of a model call that the episode gave up on: the HTTP status that the Model
    Server answered, or that the call got no reply."""
    import openai

    status = getattr(error, "status_code", None)
    return {
        "status_code": status if isinstance(status, int) else None,
        "no_reply": isinstance(error, openai.APIConnectionError),
        "error": f"{type(error).__name__}: {error}"[:2000],
    }


def call_tool(endpoint: dict[str, Any], arguments: dict[str, Any]) -> str:
    """POST ``arguments`` to an Environment Server's tool and return its reply: a JSON string's text, or the JSON."""
    headers = {"Content-Type": "application/json", **endpoint.get("headers", {})}
    if cookies := endpoint.get("cookies"):
        headers["Cookie"] = "; ".join(f"{name}={value}" for name, value in cookies.items())
    request = urllib.request.Request(
        endpoint["url"], data=json.dumps(arguments).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=TOOL_TIMEOUT_SECONDS) as response:
        reply = response.read().decode(errors="replace")
    try:
        output = json.loads(reply)
    except json.JSONDecodeError:
        return reply
    return output if isinstance(output, str) else reply


def extra_tools(task: dict[str, Any], calls: list[dict[str, Any]]) -> list[Any]:
    """The task's Environment Server tools, as Auto Ontology's extra tools that record each call in ``calls``."""
    if not task.get("tools"):
        return []
    # Only an episode with such tools needs an Auto Ontology that takes them.
    from auto_ontology.retrieval.text_to_sql.react.loop import ExtraTool

    def runner(tool: dict[str, Any]) -> Any:
        name = tool["definition"]["function"]["name"]

        def run(**arguments: Any) -> str:
            reply = call_tool(tool, arguments)
            calls.append({"name": name, "arguments": arguments, "reply": reply})
            return reply

        return run

    return [ExtraTool(definition=tool["definition"], run=runner(tool)) for tool in task["tools"]]


def run(task: dict[str, Any]) -> dict[str, Any]:
    """Answer the task's question with one ReAct episode, as Auto Ontology's ReactSQLAgent does."""
    os.environ.update(episode_environment(task))

    from auto_ontology.connectors import get_connectors
    from auto_ontology.retrieval.text_to_sql.react import loop
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
    # The loop ends with "model_error" and keeps only the text of the model call it gave up on: keep the error itself.
    failures: list[BaseException] = []
    invoke = loop.invoke_with_retry

    def invoke_with_retry(*args: Any, **kwargs: Any) -> Any:
        try:
            return invoke(*args, **kwargs)
        except Exception as error:
            failures.append(error)
            raise

    loop.invoke_with_retry = invoke_with_retry
    calls: list[dict[str, Any]] = []
    options = {"extra_tools": tools} if (tools := extra_tools(task, calls)) else {}
    result = run_episode(
        get_llm_client(),
        context,
        question=task["question"],
        evidence=task.get("evidence") or None,
        dialect=getattr(connector_for(context), "dialect", None),
        custom_prompts=task.get("custom_prompts") or None,
        **options,
    )
    submission = result.submission
    sql = submission.sql if submission is not None and submission.db_response is not None else None
    submit = task.get("submit")
    return {
        "sql": sql,
        "reason": result.reason,
        "tool_calls": result.tool_calls,
        "trajectory": result.trajectory,
        "model_failure": model_failure(failures[-1]) if result.reason == "model_error" else None,
        "environment_tool_calls": calls,
        "submit_reply": call_tool(submit, {"sql": sql}) if submit and sql else None,
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
