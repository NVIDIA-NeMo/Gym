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

An Environment Server's tools can take part, for a benchmark such as BIRD-Interact whose user the agent asks.
``tools`` are offered to the model beside Auto Ontology's own, each called over HTTP at its ``url``; ``submit`` is where
the query Auto Ontology accepts goes, as ``{"sql": ...}``. ``custom_prompts`` extends the system prompt. RESULT_JSON
then also gets each call of those tools and the reply to the submission. A tool call the server fails or refuses
replies with the error, which the model reads as the tool's result.

An episode that raises writes RESULT_JSON with ``error`` and exits 1, so the agent reports the cause.

With ``"pipeline": "interactive"``, the turn runs Auto Ontology's interactive pipeline
(``auto_ontology.retrieval.interactive``) instead, the one auto-ontology-eval runs on BIRD-Interact: it asks the user
through the ``ask_user`` tool until it submits through ``submit``, and keeps its session in ``state_path`` for the
task's next turn.

With ``"mode": "a_interact"`` as well, the pipeline runs a whole a-Interact task, as auto-ontology-eval runs it: phase
1, its retries and the follow-up, in one call, within a BIRD-COIN budget. The Environment Server charges each tool
call; the agent keeps the same account to decide what it can still afford. Each query Auto Ontology runs on the
database goes to the server's ``execute``, which charges it, and is then read from the sandbox's own copy of the
database, so Auto Ontology gets the whole result as auto-ontology-eval's do.

The model is the policy's Model Server for both of Auto Ontology's roles, unless the task names another Model Server
for the non-reasoning one (``non_reasoning_model_base_url``, ``non_reasoning_model``).
"""

import json
import os
import pickle
import re
import sys
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from string import Template
from typing import Any, Callable


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
    # Both of Auto Ontology's model roles, on Gym's Model Servers: the policy, and for the non-reasoning role the task's
    # own Model Server when it names one. The interactive pipeline calls both.
    roles = {
        "REASONING": (task["model_base_url"], task["model"]),
        "NON_REASONING": (
            task.get("non_reasoning_model_base_url") or task["model_base_url"],
            task.get("non_reasoning_model") or task["model"],
        ),
    }
    for role, (endpoint, model) in roles.items():
        settings.update(
            {f"{role}_ENDPOINT": endpoint, f"{role}_MODEL": model, f"{role}_API_KEY": MODEL_SERVER_API_KEY}
        )
    return settings


def call_tool(endpoint: dict[str, Any], arguments: dict[str, Any]) -> str:
    """POST ``arguments`` to an Environment Server's tool and return its reply: a JSON string's text, or the JSON."""
    headers = {"Content-Type": "application/json", **endpoint.get("headers", {})}
    if cookies := endpoint.get("cookies"):
        headers["Cookie"] = "; ".join(f"{name}={value}" for name, value in cookies.items())
    request = urllib.request.Request(
        endpoint["url"], data=json.dumps(arguments).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=TOOL_TIMEOUT_SECONDS) as response:
            reply = response.read().decode(errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        return f"The tool call failed with HTTP {error.code}: {body[:2000]}"
    except (urllib.error.URLError, TimeoutError) as error:
        return f"The tool call failed: {error}"
    try:
        output = json.loads(reply)
    except json.JSONDecodeError:
        return reply
    return output if isinstance(output, str) else reply


def call_charged_tool(endpoint: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """POST ``arguments`` to an a-Interact tool and return its reply: ``observation`` and ``coins_left``.

    A call that fails raises: the task cannot go on without the server, and a retry could be charged twice.
    """
    headers = {"Content-Type": "application/json", **endpoint.get("headers", {})}
    if cookies := endpoint.get("cookies"):
        headers["Cookie"] = "; ".join(f"{name}={value}" for name, value in cookies.items())
    request = urllib.request.Request(
        endpoint["url"], data=json.dumps(arguments).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=TOOL_TIMEOUT_SECONDS) as response:
            reply = json.loads(response.read().decode(errors="replace"))
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"The tool call {endpoint['url']} failed with HTTP {error.code}: {body[:2000]}") from error
    if not isinstance(reply, dict) or "observation" not in reply:
        raise RuntimeError(f"The tool {endpoint['url']} replied without an observation: {str(reply)[:500]}")
    return reply


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
    if task.get("pipeline") == "interactive":
        return run_a_interact(task) if task.get("mode") == "a_interact" else run_interactive(task)
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
        "environment_tool_calls": calls,
        "submit_reply": call_tool(submit, {"sql": sql}) if submit and sql else None,
    }


# The note the bird_interact server adds to each answer of its simulated user; the pipeline gets the answer alone.
_QUESTIONS_LEFT_NOTE = "\n\n[SYSTEM NOTE: Clarification turns remaining: "
_PHASE_PASSED = re.compile(r"Phase (\d) correct! \(Reward: ([\d.]+)\)")
# The session's objects that hold connections, which each turn makes again.
_LIVE_FIELDS = ("data_retriever", "semantic_retriever", "connectors")


def user_answer(reply: str) -> str:
    """The simulated user's answer, without the server's note on how many questions are left."""
    return reply.split(_QUESTIONS_LEFT_NOTE, 1)[0].strip()


def submission_result(reply: str) -> dict[str, Any]:
    """The submission reply as the interactive pipeline reads a submission's result: reward, phase and message."""
    passed = _PHASE_PASSED.search(reply)
    return {
        "reward": float(passed.group(2)) if passed else 0.0,
        "phase_completed": int(passed.group(1)) if passed else None,
        "message": reply,
    }


def without_database_qualifier(sql: str) -> str:
    """``sql`` with no ``<database>.`` before its schemas, which the grader's copy of the database would refuse.

    Auto Ontology's SQL generator qualifies each table with its source database. auto-ontology-eval removes it the
    same way before it submits. SQL that does not parse goes as it is.
    """
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, dialect="postgres")
        tables = [table for table in tree.find_all(exp.Table) if table.catalog]
        if not tables:
            return sql
        for table in tables:
            table.set("catalog", None)
        return tree.sql(dialect="postgres")
    except Exception:
        return sql


def run_interactive(task: dict[str, Any]) -> dict[str, Any]:
    """One c-Interact turn of Auto Ontology's interactive pipeline: questions to the user, then one submission."""
    os.environ.update(episode_environment(task))

    from auto_ontology.connectors import get_connectors
    from auto_ontology.retrieval.interactive import (
        AskUserAction,
        SubmitSQLAction,
        TurnType,
        apply_submit_result,
        apply_user_answer,
        create_session,
        step,
    )
    from auto_ontology.utils import get_data_objects_retriever, get_semantic_objects_retriever

    database = task["database"]
    connectors = {connector.database_name: connector for connector in get_connectors()}
    if database not in connectors:
        raise KeyError(f"No source database {database!r}; the catalog has {sorted(connectors)}")
    data_retriever = get_data_objects_retriever()
    live = {
        "data_retriever": data_retriever,
        "semantic_retriever": get_semantic_objects_retriever() or data_retriever,
        "connectors": [connectors[database]],
    }
    state_path = Path(task["state_path"])
    if state_path.is_file():
        session = pickle.loads(state_path.read_bytes())
        for name, value in live.items():
            setattr(session, name, value)
    else:
        session = create_session(
            session_id=state_path.parent.name,
            task_id=database,
            db_name=database,
            db_schema=task["schema"],
            external_kb=task["knowledge"],
            question="",
            max_clarify_turns=task["max_questions"],
            external_kb_children_map=task.get("knowledge_children") or {},
            # As auto-ontology-eval creates the session for BIRD-Interact.
            skip_evidence_refinement=True,
            numeric_type_alignment=True,
            **live,
        )
    turn = task["turn"]
    turn_type, options = {
        "phase_1": (TurnType.INITIAL, {"initial_question": turn["text"]}),
        "debug": (TurnType.DEBUG, {"debug_error": turn.get("execution_error")}),
        "follow_up": (TurnType.FOLLOW_UP, {"follow_up_question": turn["text"]}),
    }[turn["kind"]]
    ask = next(tool for tool in task["tools"] if tool["definition"]["function"]["name"] == "ask_user")
    calls: list[dict[str, Any]] = []
    sql, reply, reason, steps = None, None, "step_limit", 0
    while steps < task["max_steps"]:
        steps += 1
        action = step(session, turn_type=turn_type, **options)
        if isinstance(action, AskUserAction):
            answer = call_tool(ask, {"question": action.question})
            calls.append({"name": "ask_user", "arguments": {"question": action.question}, "reply": answer})
            apply_user_answer(session, user_answer(answer))
            continue
        if isinstance(action, SubmitSQLAction):
            sql = without_database_qualifier(action.sql)
            reply = call_tool(task["submit"], {"sql": sql})
            apply_submit_result(session, submission_result(reply))
            reason = "submitted"
        else:
            reason = f"unexpected action {type(action).__name__}"
        break
    for name in _LIVE_FIELDS:
        setattr(session, name, None)
    state_path.write_bytes(pickle.dumps(session))
    return {
        "sql": sql,
        "reason": reason,
        "tool_calls": steps,
        "trajectory": None,
        "environment_tool_calls": calls,
        "submit_reply": reply,
    }


# ----- a-Interact ----- #
#
# The rules and texts below are auto-ontology-eval's (ontology_sql_eval/bird_interact/budget.py and server.py of
# https://github.com/NVIDIA/auto-ontology-eval at e727e4d6e1e198355099de4803590bb3681793a4, Apache-2.0), which follow the
# BIRD-Interact ADK's a-Interact callbacks.

# The BIRD-COIN price of each tool the episode calls, as the bird_interact server charges them.
COIN_COSTS = {"execute": 1.0, "ask": 2.0, "submit": 3.0}
# Phase 1's queries and questions leave this many coins. A submission can spend them, and phase 2 has no floor.
PHASE_1_FLOOR = 4.0
# What a phase-2 attempt needs: a submission, the query that checks it, and two to look at the data first.
PHASE_2_RESERVE = 6.0
# A task's steps, whatever its budget: the ADK agent's MAX_MODEL_TURNS.
MAX_A_INTERACT_STEPS = 60
# The simulated user's reply when it has no answer; after it the agent asks no more.
USER_FAILED_REPLY = "I'm not sure I understand your question."
# The server's reply to a call once the budget is spent.
BUDGET_SPENT_REPLY = "Your budget is spent. Only submit is allowed now."
# auto-ontology-eval's instructions after a follow-up question, which extend phase 1's accepted query.
FOLLOW_UP_GUIDANCE = """\
Execution requirements:
- Treat the accepted Phase 1 SQL as authoritative for the prior result set, formulas, joins, filters, ranking, and tie behavior. Preserve every part the follow-up does not explicitly replace.
- Apply only the requested change. References such as "the same", "those", "top N", or "from that list" must use the Phase 1 result as the base relation before extending it.
- Return exactly the requested columns, in the requested order and grain, with no extra columns. Preserve requested rounding, scale, NULL handling, and sorting."""
_FOLLOW_UP_LINE = "\nFollow-up question: "
_FAILED = re.compile(r"\ASQL failed Phase \d\. ")
# The grader's messages for a submission that did not run (environment_servers/bird_interact/adk_prompts.py).
_EXECUTION_FAILURES = ("Error executing submitted SQL: ", "Submitted SQL execution timed out")
_TASK_ENDED = "\nYour budget is spent, so the task ends here."


class BudgetExhausted(RuntimeError):
    """A query the budget cannot pay for; the server would refuse it too."""


class Budget:
    """The task's BIRD-COIN, as the agent accounts them; the server's own account decides what it charges."""

    def __init__(self, initial: float, remaining: float) -> None:
        self.initial, self.remaining = initial, remaining
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        # The ADK's stop signal, never a real balance.
        return self.remaining < 0

    def charge(self, tool: str, *, reserve: float = 0.0) -> bool:
        """Spend ``tool``'s price, or return False when the call is refused.

        A submission is never refused: one the balance cannot cover, or that spends it to zero, is the task's last
        action, and the budget reads exhausted after it. Any other tool is refused when it would leave less than
        ``reserve``, or more than the balance.
        """
        with self._lock:
            cost = COIN_COSTS[tool]
            if tool != "submit" and self.remaining - cost < reserve:
                return False
            if self.remaining < cost:
                if tool != "submit":
                    return False
                self.remaining = -1.0
                return True
            self.remaining -= cost
            if tool == "submit" and self.remaining <= 0:
                self.remaining = -1.0
            return True

    def affordable_questions(self, reserve: float = 0.0) -> int:
        """Questions that still leave one submission, or ``reserve`` if that is more."""
        spare = self.remaining - max(reserve, COIN_COSTS["submit"])
        return max(0, int(spare // COIN_COSTS["ask"]))


class ChargedConnector:
    """A database connector whose queries the server charges as ``execute``, before they run on this sandbox's copy.

    A query the budget cannot pay for raises ``BudgetExhausted`` without running. Everything else (schema
    introspection and the like) passes to the connector free, as the schema was bought once. ``cost_status`` and
    ``can_run_queries`` tell Auto Ontology's ReAct agent what it can still afford.
    """

    def __init__(
        self,
        base: Any,
        budget: Budget,
        execute: dict[str, Any],
        calls: list[dict[str, Any]],
        phase: Callable[[], int],
    ) -> None:
        self._base, self._budget, self._execute, self._calls, self._phase = base, budget, execute, calls, phase
        # Phase 1 keeps its floor; the follow-up clears it.
        self.reserve = PHASE_1_FLOOR
        self._lock = threading.Lock()

    def cost_status(self) -> str:
        left = self._budget.remaining
        if not self.can_run_queries():
            return "BUDGET: no coins left for run_sql. Submit your best query now."
        prices = (
            f"run_sql costs {COIN_COSTS['execute']:g} coin. Submitting costs {COIN_COSTS['submit']:g}, plus 1 for "
            "each query run to check the SQL, and a wrong submission is charged in full."
        )
        if self._phase() == 1:
            return (
                f"BUDGET: {left:g} coins left. {prices} If this answer is accepted, a follow-up question comes next "
                f"and is paid from what is left, so submit with at least {PHASE_2_RESERVE:g} coins unspent when you can."
            )
        return f"BUDGET: {left:g} coins left for this follow-up. {prices}"

    def can_run_queries(self) -> bool:
        return self._budget.remaining - COIN_COSTS["execute"] >= self.reserve

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            charged = self._budget.charge("execute", reserve=self.reserve)
            reply = call_charged_tool(self._execute, {"sql": sql}) if charged else None
            self._calls.append(
                {
                    "name": "execute",
                    "arguments": {"sql": sql[:2000]},
                    "charged": charged,
                    "coins_left": None if reply is None else reply.get("coins_left"),
                }
            )
        if not charged or (reply is not None and reply["observation"] == BUDGET_SPENT_REPLY):
            raise BudgetExhausted(f"Query refused ({self._budget.remaining:.1f} remaining).")
        return self._base.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def follow_up_question(reply: str) -> str | None:
    """The follow-up question in a submission's reply, with auto-ontology-eval's instructions; None without one."""
    if _FOLLOW_UP_LINE not in reply:
        return None
    question = reply.split(_FOLLOW_UP_LINE, 1)[1].strip()
    return f"{question}\n\n{FOLLOW_UP_GUIDANCE}" if question else question


def a_interact_execution_error(reply: str) -> str | None:
    """A failed submission's error when it did not run, as the grader wrote it; None when it ran but was wrong."""
    if not _FAILED.match(reply):
        return None
    message = _FAILED.sub("", reply, count=1).removesuffix(_TASK_ENDED)
    return message if message.startswith(_EXECUTION_FAILURES) else None


def run_a_interact(task: dict[str, Any]) -> dict[str, Any]:
    """A whole a-Interact task in Auto Ontology's interactive pipeline, as auto-ontology-eval runs it."""
    os.environ.update(episode_environment(task))

    from auto_ontology.connectors import get_connectors
    from auto_ontology.retrieval.interactive import (
        AskUserAction,
        SubmitSQLAction,
        TurnType,
        apply_submit_result,
        apply_user_answer,
        create_session,
        step,
    )
    from auto_ontology.utils import get_data_objects_retriever, get_semantic_objects_retriever

    database = task["database"]
    connectors = {connector.database_name: connector for connector in get_connectors()}
    if database not in connectors:
        raise KeyError(f"No source database {database!r}; the catalog has {sorted(connectors)}")
    tools = task["tools"]
    budget = Budget(initial=float(task["budget"]["initial"]), remaining=float(task["budget"]["remaining"]))
    calls: list[dict[str, Any]] = []
    phase = 1
    connector = ChargedConnector(connectors[database], budget, tools["execute"], calls, lambda: phase)
    data_retriever = get_data_objects_retriever()
    session = create_session(
        session_id=f"{database}-a-interact",
        task_id=database,
        db_name=database,
        db_schema=task["schema"],
        external_kb=task["knowledge"],
        question="",
        data_retriever=data_retriever,
        semantic_retriever=get_semantic_objects_retriever() or data_retriever,
        connectors=[connector],
        max_clarify_turns=budget.affordable_questions(PHASE_1_FLOOR),
        external_kb_children_map=task.get("knowledge_children") or {},
        # As auto-ontology-eval creates the session for BIRD-Interact.
        skip_evidence_refinement=True,
        numeric_type_alignment=True,
    )

    turn_type, debug_error, follow_up = TurnType.INITIAL, None, None
    submitted: set[str] = set()
    questions_blocked = False
    sql, reply, reason, steps, coins_left = None, None, "max_agent_steps", 0, None

    def next_action() -> Any:
        return step(
            session,
            turn_type=turn_type,
            debug_error=debug_error,
            initial_question=task["question"],
            follow_up_question=follow_up,
        )

    while steps < MAX_A_INTERACT_STEPS:
        steps += 1
        if phase == 1:
            # The pipeline asks while its history is under max_clarify_turns: as many as the budget can still pay.
            questions = 0 if questions_blocked else budget.affordable_questions(PHASE_1_FLOOR)
            session.max_clarify_turns = len(session.clarify_history) + questions
        action = next_action()
        if isinstance(action, AskUserAction) and questions_blocked:
            session.max_clarify_turns = len(session.clarify_history)
            action = next_action()
            if isinstance(action, AskUserAction):
                reason = "user_sim_failed"
                break
        if isinstance(action, AskUserAction):
            if not budget.charge("ask", reserve=PHASE_1_FLOOR if phase == 1 else 0.0):
                reason = "ask_unaffordable"
                break
            answered = call_charged_tool(tools["ask"], {"question": action.question})
            answer, coins_left = answered["observation"], answered.get("coins_left")
            if answer.strip() == USER_FAILED_REPLY:
                questions_blocked = True
            calls.append({"name": "ask", "arguments": {"question": action.question}, "reply": answer})
            apply_user_answer(session, user_answer(answer))
            continue
        if not isinstance(action, SubmitSQLAction):
            reason = f"unexpected_action:{type(action).__name__}"
            break
        if not action.sql.strip():
            recovered = str((getattr(session, "path_state", None) or {}).get("sql_code") or "").strip()
            if not recovered:
                reason = "empty_sql"
                break
            action = SubmitSQLAction(recovered)
        if action.sql in submitted:
            # The grader is deterministic: a query that failed fails again.
            reason = "repeated_sql"
            break
        submitted.add(action.sql)
        budget.charge("submit")
        sql = without_database_qualifier(action.sql)
        graded = call_charged_tool(tools["submit"], {"sql": sql})
        reply, coins_left = graded["observation"], graded.get("coins_left")
        calls.append({"name": "submit", "arguments": {"sql": sql}, "reply": reply})
        result = submission_result(reply)
        apply_submit_result(session, result)
        if result["phase_completed"] == 1:
            question = follow_up_question(reply)
            if question is None:
                reason = "done"
                break
            phase, submitted, connector.reserve = 2, set(), 0.0
            turn_type, follow_up, debug_error = TurnType.FOLLOW_UP, question, None
        elif result["phase_completed"] == 2:
            reason = "done"
            break
        else:
            turn_type, debug_error = TurnType.DEBUG, a_interact_execution_error(reply)
        if budget.exhausted:
            reason = "budget_exhausted"
            break
    return {
        "sql": sql,
        "reason": reason,
        "tool_calls": steps,
        "trajectory": None,
        "environment_tool_calls": calls,
        "submit_reply": reply,
        # The agent's account and the server's last word on it, which agree unless the two rules part.
        "budget": {"initial": budget.initial, "remaining": budget.remaining, "server_coins_left": coins_left},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    task_path, result_path = (Path(arg) for arg in argv)
    try:
        result = run(json.loads(task_path.read_text()))
    except Exception as error:
        traceback.print_exc()
        result_path.write_text(json.dumps({"error": f"{type(error).__name__}: {error}"[:4000]}))
        return 1
    result_path.write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
