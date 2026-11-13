# BIRD-Interact Environment Server

[BIRD-Interact](https://bird-interact.github.io/) ([paper](https://arxiv.org/abs/2510.05318)) is an interactive
text-to-SQL benchmark. The request is ambiguous, and some knowledge-base entries are hidden. The agent must ask a
simulated user to resolve the ambiguity before it writes SQL.

This Environment Server runs both of its modes with no Resources Server. It is a `BaseEnvironmentServerWithTools`,
which serves the episode's tools. Its `run` seeds the agent session (`agent_session.py`), runs the turns and
grades them. Each episode holds one task:

- a sandbox with the task's database, with `preprocess_sql` applied (see [Database sandbox](#database-sandbox));
- the two-stage user simulator, which calls the `user_model` server;
- the grader, the current phase, the SQL attempts and the reward;
- the tools: two in c-Interact, nine in a-Interact with their BIRD-COIN budget.

c-Interact follows the [BIRD-Interact ADK](https://github.com/bird-bench/BIRD-Interact/tree/451fe2c3518ee1cf908d8139e2913483bd519381/BIRD-Interact-ADK)
(revision `451fe2c`): its protocol, its user simulator and its grading, for the query and management tasks of Lite and
Full. [ADK parity](#adk-parity) lists what matches and the one deliberate difference.

## Modes

| `mode` | Turns | Agent's tools | Next turn |
|---|---|---|---|
| `c_interact` | Up to four: each phase, and a debug turn after a failed one. The agent must resume its session across turns. | `ask_user` and `submit_sql`, the ADK's | The ADK orchestrator's message: the debug message, the follow-up question, or none. |
| `a_interact` | One. | Nine: `ask`, `submit`, `execute`, `get_schema`, `get_all_column_meanings`, `get_column_meaning`, `get_all_external_knowledge_names`, `get_knowledge_definition`, `get_all_knowledge_definitions` | None (`agent_finished`). |

The server serves the tools for the episode over each of `tool_transports`. Over `direct_http` the first request
declares them as function tools; over `mcp` alone it declares none, because the agent lists them from the server and
Codex refuses a request that declares tools.

**Tools.** The agent calls the tools with a signed episode token that its seed grants. Over `direct_http` it calls
`POST /<tool name>` on this server; over `mcp`, `/mcp` lists only its episode's tools. The token works only on the
tool routes, for this episode, until the episode ends. With several workers, each call reaches the worker that runs
its episode. The tools have the same names, descriptions and arguments over both transports.

**c-Interact, as in the ADK.** The first request is the ADK's instruction (the database name, its schema, the visible
knowledge base and the clarification turns, `max_turn`) and its phase-1 message (the ambiguous request). `max_turn`
is the task's critical and knowledge ambiguities plus `question_patience` (default 3, the ADK's patience).

- `ask_user` answers through the simulator and adds `[SYSTEM NOTE: Clarification turns remaining: N/max_turn]`. As in
  the ADK, questions are not capped; the instruction tells the agent to submit after `max_turn`.
- `submit_sql` grades the SQL for the current phase and returns the grader's message, the reward and the follow-up
  question when they apply, and `Budget remaining: 0 bird-coins`, as the ADK's tool does.
- A turn ends at its submission: a later call in the turn gets `SQL submitted. Awaiting result.`, and a call once the
  task is done `Task completed.`, the replies the ADK's agent gets.
- After a turn in which the phase did not pass, with a submission or without one, the next turn is the phase's debug
  turn: `Your SQL is not executable: <error>` when the last submission did not run, `Your SQL is not correct. You have
  one more chance.` otherwise. After phase 1 passes, the follow-up question opens phase 2, which has its own debug turn.
- For SQLite, the ADK's texts say SQLite where they say PostgreSQL.

**a-Interact rules.** Each action costs BIRD-COIN (`action_costs`). The budget is 6, plus the cost of one `ask` for
each ambiguity, plus `coin_patience` (default 6). When the budget is spent, only `submit` runs. A failed `submit`
with no budget left ends the task. `submit` replies with the grader's message, as in c-Interact.

**Rewards.** `phase_rewards` has one entry for each phase, with the reward for a pass on the phase's first submission
and on a later one. The default is the ADK's: phase 1 0.7 / 0.5, phase 2 0.3 / 0.2. As in the ADK, it counts
submissions, not turns: a debug turn's submission earns the first-submission reward when the phase's first turn did
not submit. a-Interact always uses the first value. Mini-Interact has no follow-up and scores 1.0 / 0.5:

```yaml
phase_rewards:
  - {first_attempt: 1.0, after_debug: 0.5}
```

**Result.** `reward`, `turns`, `stop_reason`, `tool_calls` and the agent's observations, plus `phases`,
`questions_asked`, `coins_spent` and `finished`. Stop reasons: `solved`, `failed` (the debug turn did not pass),
`no_sql_submitted` (neither of the phase's turns submitted), `agent_finished` (a-Interact), `interaction_timeout`,
and the failure kinds below.

## User simulator

The ADK's two-stage simulator, with its default (`v2`) prompts. The action parser reads the agent's question, the
task's annotated ambiguities and the solution SQL split into clauses, and picks one action: `labeled("<term>")`,
`unlabeled("<segment>")` or `unanswerable()`. The response generator then writes the user's reply from that action,
the clear question, the solution SQL, its clauses and the schema. In phase 2 the prompts show no ambiguities and the
follow-up's solution, and keep phase 1's clear question. Both stages run at temperature 0, with at most 500 and 1024
output tokens, and read the model's output as the ADK does: the action and the reply between `<s>` and `</s>`, and
`I'm not sure I understand your question.` when the reply has neither.

The simulator model is the `user_model` server, which `user_model_name` in `env.yaml` sets. The ADK's default, which
auto-ontology-eval uses, is `anthropic/claude-haiku-4-5-20251001`; set it for scores comparable with theirs.
BIRD-Interact's own harness, `bird_interact_agent`, uses `gemini-2.0-flash-001`.

If the user model fails, the result masks the sample with `failure_kind: bird_interact:user_simulator_failed`, and
c-Interact opens no further turn.

## Database sandbox

The episode starts a sandbox and stops it when the episode ends, on every path; at shutdown the server finishes
the cleanups of episodes still running, and `sandbox_config.ttl_s` lets the provider reclaim a sandbox that a
crashed server left. The server uploads `grading.py`, and for SQLite the database source, and runs
`python3 grading.py <setup|execute|grade> <request.json>` there. For SQLite the image needs only `python3` (default
`python:3.12-slim`); for PostgreSQL see [PostgreSQL (Lite and Full)](#postgresql-lite-and-full). The agent's SQL and the grading run in the sandbox, never in the server process, and the agent
never gets the sandbox itself. `grading.py` deletes each request file once it has read it, so the solution SQL in a
grade request does not stay in the sandbox.

`sandbox_provider: sandbox` names the provider block that a provider config binds, such as
`nemo_gym/sandbox/providers/docker/configs/docker.yaml`. The local provider
(`nemo_gym/sandbox/providers/local/configs/local.yaml`) runs the commands on the host, which the tests use; the
server then logs a warning at startup.

A command that the sandbox does not run, such as a provider error, a timeout or a crash of `grading.py`, is an
infrastructure failure, not the agent's mistake. It runs again on a fresh sandbox up to
`inconclusive_verification_retries` times (default 1); the fresh sandbox's setup applies phase 1's SQL too, once it
has passed. If it still fails, the result masks the sample with `failure_kind: bird_interact:sandbox_failed`, and
c-Interact opens no further turn.

## Grading

`grading.py` is an independent implementation of the ADK's default test case, which grades every query phase. A
submission runs as the ADK runs it, in order on one database state: the submitted SQL as written, which shows whether
it runs (an error or a timeout is the debug message's `<error>`); then the submitted SQL and the solution's, each with
its comments, `DISTINCT` words and `ROUND(...)` calls removed. The results are compared: floats and decimals to two
places (decimals half-up), dates and timestamps as `YYYY-MM-DD`, JSON values and arrays as sorted JSON; rows in order
when `conditions.order` is set and as sets otherwise. Each query keeps the rows of its last statement, at most 10,000
when it starts with `SELECT` or `WITH`, and each statement times out after `sql_timeout_seconds` (default 60).

Nothing that grading runs stays in the database: SQLite runs it on a scratch copy, PostgreSQL in a transaction that it
rolls back. When phase 1 passes and phase 2 follows, its SQL is applied, so phase 2 sees what it changed, as the ADK's
phase-1 snapshot does.

The ADK's rules, kept for parity, include its edge cases: a comment at the very end, with no line break after it,
stays; `COUNT(DISTINCT x)` keeps its `DISTINCT`; an array of decimals or dates fails the comparison, because it cannot
be written as JSON; and a submission that writes runs twice, so a write that cannot run twice fails.

A query phase is graded as above even when it has test cases. A phase whose `category` is not `Query` and that has
`test_cases`, a management task's, is graded by its test cases, as in the ADK: Python code from the ground truth that
defines `test_case(pred_sqls, sol_sqls, db_name, conn)` and raises when the database is wrong. `grading.py` runs it in
the episode's sandbox, never on the Gym host, on PostgreSQL only: with SQLite (Mini-Interact), the server refuses
such a task at seed.

- The phase runs on a copy of the episode's database, which `grading.py` drops afterwards. The submission runs first,
  and its writes stay on the copy for the test cases. An error or a timeout is the debug message's `<error>`.
- Each test case gets the ADK's six globals: `execute_queries` (each query committed, under the statement timeout;
  it returns the rows of the last, an error, and whether it timed out), `ex_base` (the default test case's
  comparison, 1 or 0), `remove_comments`, `remove_distinct` and `remove_round` (on lists of SQL), and
  `pred_query_result` (the rows the submission returned).
- The test cases run in order. The first that raises anything fails the phase. As in the ADK, the code's globals
  are those six names and its locals its own, so a function in it does not see the code's top-level imports.
- A sandbox command has time for the submission and two queries for each test case, each `sql_timeout_seconds`.

In Lite and Full, only the management tasks have such a phase; every query task's follow-up is a query.

## Harnesses

What each harness's Gym adapter accepts today, read from its session seed checks. c-Interact needs both a
multi-turn session and tool grants. `simple_agent`, in both modes, and Codex, in a-Interact over `mcp`, have run BIRD
with a model; the other rows are not results from runs.

| Harness | c-Interact | a-Interact |
|---|---|---|
| `simple_agent` | Yes, over `direct_http` | Yes, over `direct_http` |
| Codex, Pi, Hermes | No: the adapter has no multi-turn sessions | Yes, over `mcp`, in the harness's own sandbox |
| Stirrup | No: no multi-turn sessions | Yes, over `direct_http`, in its own sandbox. `code_exec: false` leaves out its shell. |
| OpenCode | No: it refuses required tool grants | No: it refuses required tool grants |
| Claude Code, Kilo, OpenClaw | No: no multi-turn sessions | No: they refuse required tool grants, MCP included |

- **Neither mode grants a sandbox.** BIRD's sandbox holds the database and the ground truth. A sandboxed harness
  runs in its own sandbox, from the `sandbox_provider` in the agent's config. Turn off its built-in tools where the
  adapter allows it, so scores stay comparable with the paper's.
- **Over `mcp`, the harness calls this server from inside its sandbox**, so the server's address must be reachable
  from there. To run a-Interact on Codex or Pi, point `agent_server` at the harness and set
  `tool_transports: [mcp]`. `benchmarks/gdpval/codex.yaml` and `benchmarks/gdpval/pi.yaml` show the agent settings.
- **With the Docker provider**, start the servers with `+default_host` set to an address the container can reach,
  such as the `docker0` bridge address. If the host's firewall drops traffic from the bridge to host ports, give the
  harness's sandbox provider `network: host`. BIRD's database sandbox needs neither, because the server runs
  commands in it and it calls nothing.

## Data

`database_dir` holds one directory for each database, in the BIRD-Interact layout:

```text
<db>/<db>.sqlite (or <db>.sql), for SQLite only
<db>/<db>_schema.txt
<db>/<db>_column_meaning_base.json
<db>/<db>_kb.jsonl
```

`data/example.jsonl` has five tasks on `data/example_databases/bookshop`, a synthetic database written for this
server. `bookshop_3` and `bookshop_4` have a follow-up question. `bookshop_4` and `bookshop_5` have two critical
ambiguities each, as most published tasks do, and `bookshop_5` hides no knowledge. `task_data.py` is the row schema.

The agent's config declares the file once for each mode's taskset, `bird_interact:c_interact_example` and
`bird_interact:a_interact_example`, and `environment_server_routes` sends each to its mode's server.
`data/example_rollouts.jsonl` has one rollout for each task in each mode, with `simple_agent`.

The public BIRD-Interact releases (Mini-Interact, Lite, Full) withhold `sol_sql` and `test_cases`. The authors
send them on request. A row without `sol_sql` fails the episode at seed, for good. Mini-Interact uses SQLite, so its
databases work as downloaded from [`birdsql/mini-interact`](https://huggingface.co/datasets/birdsql/mini-interact),
after the ground truth is merged into the rows. Lite and Full use PostgreSQL.

## PostgreSQL (Lite and Full)

With `database_engine: postgresql`, a task's database is `<db>_template` in the PostgreSQL server of the sandbox
image, and each episode starts its own sandbox from that image. `setup` starts the server and applies
`preprocess_sql`; after that, each query runs in a transaction that is rolled back, except phase 1's passing SQL, so a
write in one query or episode never reaches another. Each query is sent whole, as the ADK sends it, so it can hold
several statements and dollar-quoted bodies. The image holds the release's databases already restored, so an episode
starts PostgreSQL instead of restoring dumps.

[`benchmarks/bird_interact`](../../benchmarks/bird_interact/README.md) builds the image, prepares the Lite or Full
data with the ground truth, and runs the benchmark.

## ADK parity

What c-Interact takes from the ADK at revision `451fe2c3518ee1cf908d8139e2913483bd519381`, and where it lives:

| ADK | Gym |
|---|---|
| `orchestrator/cinteract.py`: `max_turn`, the phase-1, debug and follow-up messages, two turns for each phase | `app.py` (`_first_request`, `_next_message`) |
| `system_agent/agent.py`, `tools.py`, `callbacks_cinteract.py`: the instruction, `ask_user`, `submit_sql`, one submission for each turn | `app.py` (`CInteractTools`) |
| `db_environment/server.py`: the grader's messages, the rewards by submission, the phase-1 snapshot, the knowledge the agent sees | `app.py` (`submit`), `grading.py` |
| `shared/db_utils.py`: `test_case_default`, its rewrites, and the helpers of the management test cases | `grading.py` |
| `user_simulator/`: the two stages, the `v2` prompts, the clause split | `simulator.py` |

The texts (prompts, instruction, tool descriptions and messages) are the ADK's, verbatim, in `adk_prompts.py`; the
code is Gym's. Checks back the grading: the tests, and runs of the ADK's own code against `grading.py` in the Lite
image. For queries, `test_case_default` on six Lite databases gave the same verdict on 863 of 863 submissions. For
management phases, the ADK's submit flow and its helpers gave the same verdict in 306 of 306 comparisons: 17
scenarios with test cases written for the check (writes, views, functions, errors, imports, the helpers, and one task
across both phases) on each of the 18 Lite databases. Each time, the episode's database was unchanged afterwards.
The real management test cases arrive with the ground truth (GYMINT-685).

The ADK's original evaluation (`evaluation/src`) gives a test case more names than the ADK does, such as
`check_sql_function_usage` and `performance_compare_by_qep`. Gym gives the ADK's six; a test case that calls another
fails, as in the ADK.

The ADK's agent-side behavior that a Gym harness replaces is not reproduced: its agent stops at 60 model calls in an
episode, where a Gym harness has its own limit (`simple_agent`'s `max_steps`, for each turn). The one deliberate
difference is in the simulator: the ADK fills its prompts' placeholders one after another, so a question containing
`[[GT_SQL]]` would get the solution SQL written into it. Gym fills them in one pass, so it cannot.

## Not supported yet

- The data and catalog checks that `gym env test` runs for a Resources Server (GYMINT-643).

## Running

`env.yaml` needs `policy_base_url`, `policy_api_key` and `policy_model_name`, and the simulator's
`user_model_base_url`, `user_model_api_key` and `user_model_name`. Start the servers with a sandbox provider:

```bash
gym env start \
    --config environment_servers/bird_interact/configs/bird_interact.yaml \
    --model-type vllm_model \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml
```

Rows route to an Environment Server by taskset, so write each example row as a materialized task, once for each
mode, and run them against the started servers:

```bash
python3 - <<'EOF'
import json
rows = [json.loads(line) for line in open("environment_servers/bird_interact/data/example.jsonl")]
with open("bird_interact_example_input.jsonl", "w") as f:
    for mode in ("c_interact", "a_interact"):
        for row in rows:
            task_id = {"taskset": f"bird_interact:{mode}_example", "task_id": row["instance_id"]}
            f.write(json.dumps({"task_id": task_id, "task_input": row}) + "\n")
EOF

gym eval run --no-serve \
    --config environment_servers/bird_interact/configs/bird_interact.yaml \
    --model-type vllm_model \
    --input bird_interact_example_input.jsonl \
    --output results/bird_interact_example.jsonl
```

## Licensing

Code: Apache 2.0. `adk_prompts.py` holds the BIRD-Interact ADK's texts, under its MIT license (in the file's
header; see the repository's `ATTRIBUTIONS.md`). The example database and tasks: Apache 2.0. BIRD-Interact data: CC
BY-SA 4.0.
