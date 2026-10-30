# BIRD-Interact Resources Server

Episode state for [BIRD-Interact](https://bird-interact.github.io/) ([paper](https://arxiv.org/abs/2510.05318)),
an interactive text-to-SQL benchmark. The request is ambiguous, and some knowledge-base entries are hidden. The agent
must ask a simulated user to resolve the ambiguity before it writes SQL.

One session holds one task:

- a sandbox with the task's SQLite database, with `preprocess_sql` applied (see [Session sandbox](#session-sandbox));
- the two-stage user simulator, which calls the `user_model` server;
- the grader, the current phase, the SQL attempts and the reward;
- in a-Interact, the BIRD-COIN budget.

The `bird_interact` Environment Server (`environment_servers/bird_interact`) runs the protocol in both modes on
`BaseUserSimulationEnvironmentServer`.

## Modes and routes

| Mode | Who owns the loop | Routes |
|---|---|---|
| c-Interact | The Environment Server. Each agent reply is a question (`/c_interact/ask`) or SQL (`/c_interact/submit`). | `/c_interact/ask`, `/c_interact/submit`, `/verify` |
| a-Interact | The agent, in one turn. | Nine tools at the root, and over MCP at `/mcp`: `ask`, `submit`, `execute`, `get_schema`, `get_all_column_meanings`, `get_column_meaning`, `get_all_external_knowledge_names`, `get_knowledge_definition`, `get_all_knowledge_definitions` |

The seed reply carries the context for the Environment Server's `responses_create_params`. In c-Interact it has the schema,
the column meanings, the visible knowledge base and the question budget. In a-Interact it has only the database, the
request and the BIRD-COIN budget, because the tools charge for the rest.

Both modes use the same simulator and grader. Each mode's routes refuse a session of the other mode, so an
a-Interact agent cannot submit through `/c_interact/submit` without paying for it.

**MCP.** With `expose_tools_over_mcp: true`, as in the config, the seed reply also carries `resources_tools`: the
`/mcp` endpoint and a session token. The token lists the nine tools for an a-Interact session and none for a
c-Interact one, and the `/c_interact/` routes are never exposed. The tools have the same names, descriptions and
arguments over direct HTTP and MCP (`A_INTERACT_TOOLS` in `schemas.py`).

**c-Interact rules.** The agent may ask as many questions as the task has critical and knowledge ambiguities, plus
`question_patience` (default 3), before its first submission. A submission that fails gets one more attempt,
which must be SQL. A task with a `follow_up` asks it as
phase 2 after phase 1 passes; phase 2 has the same two attempts.

**a-Interact rules.** Each action costs BIRD-COIN (`action_costs`). The budget is 6, plus the cost of one `ask` for
each ambiguity, plus `coin_patience` (default 6). When the budget is spent, only `submit` runs. A failed `submit`
with no budget left ends the task.

**Rewards.** `phase_rewards` has one entry for each phase, with the reward for a pass on the first attempt and after
the debug attempt. The default is the paper's: phase 1 0.7 / 0.5, phase 2 0.3 / 0.2. a-Interact always uses the
first-attempt value. Mini-Interact has no follow-up and scores 1.0 / 0.5:

```yaml
phase_rewards:
  - {first_attempt: 1.0, after_debug: 0.5}
```

## User simulator

Stage 1 classifies the agent's question as an annotated ambiguity (`labeled`), a detail that the ground truth
settles but that is not annotated (`unlabeled`), or a question to refuse (`unanswerable`). Stage 2 writes the reply
from that action only. A refused question gets a fixed reply without a model call. A term in the question that
links to a hidden knowledge entry, through the same SQL fragment, lets the reply use that entry.

The prompts are written for Gym, not copied from the BIRD repository. Parity runs against the published scores need
the reference prompts and the reference simulator model (`gemini-2.0-flash-001`).

If the user model fails during an a-Interact `ask`, `/verify` masks the sample with
`failure_kind: bird_interact:user_simulator_failed`. In c-Interact the failure fails the episode, which is retried.

## Session sandbox

The session starts a sandbox at seed and stops it at close; at shutdown the server stops any sandbox still open, and
`sandbox_config.ttl_s` lets the provider reclaim one that a crashed server left. The server uploads `grading.py` and
the database source to the sandbox and runs `python3 grading.py <setup|execute|grade> <request.json>` there, so the
image needs only `python3` (default `python:3.12-slim`). The agent's SQL and the grading run in the sandbox, never in
the server process, and the agent never gets the sandbox itself. `grading.py` deletes each request file once it has
read it, so the solution SQL in a grade request does not stay in the sandbox.

`sandbox_provider: sandbox` names the provider block that a provider config binds, such as
`nemo_gym/sandbox/providers/docker/configs/docker.yaml`. The local provider
(`nemo_gym/sandbox/providers/local/configs/local.yaml`) runs the commands on the host, which the tests use.

A command that the sandbox does not run, such as a provider error, a timeout or a crash of `grading.py`, is an
infrastructure failure, not the agent's mistake. It runs again on a fresh sandbox up to
`inconclusive_verification_retries` times (default 1). If it still fails, the route answers HTTP 503: in c-Interact
the Environment Server retries the episode, and in a-Interact `/verify` masks the sample with
`failure_kind: bird_interact:sandbox_failed`.

## Grading

`grading.py` is an independent implementation of BIRD-Interact's default test case. It removes comments, `DISTINCT`
and `ROUND(...)` from both queries, runs each on a scratch copy of the session database, rounds floats to two places,
and compares rows in order when `conditions.order` is set, or as sets otherwise. An empty result fails.

## Data

`database_dir` holds one directory for each database, in the BIRD-Interact layout:

```text
<db>/<db>.sqlite (or <db>.sql)
<db>/<db>_schema.txt
<db>/<db>_column_meaning_base.json
<db>/<db>_kb.jsonl
```

`data/example.jsonl` has five tasks on `data/example_databases/bookshop`, a synthetic database written for this
server. `bookshop_3` and `bookshop_4` have a follow-up question. `bookshop_4` and `bookshop_5` have two critical
ambiguities each, as most published tasks do, and `bookshop_5` hides no knowledge.

The config declares the file once for each mode's taskset, under one dataset name, so both share
`data/example_metrics.json`. `data/example_rollouts.jsonl` has one rollout for each task in each mode, with
`simple_agent`: the five c-Interact rollouts, then the five a-Interact ones. The data check wants five for each
taskset.

The public BIRD-Interact releases (Mini-Interact, Lite, Full) withhold `sol_sql` and `test_cases`. The authors
send them on request. A seed for a row without `sol_sql` is refused with HTTP 422. Mini-Interact uses SQLite, so its
databases work as downloaded from [`birdsql/mini-interact`](https://huggingface.co/datasets/birdsql/mini-interact),
after the ground truth is merged into the rows.

## Not supported yet

- Rows with custom `test_cases` (BIRD-Interact's management tasks). The seed refuses them with HTTP 422.
- PostgreSQL (Lite and Full).
- Database changes that carry over from phase 1 to phase 2. Each submission runs on a scratch copy.

## Running

`env.yaml` needs `policy_base_url`, `policy_api_key` and `policy_model_name`, and the simulator's
`user_model_base_url`, `user_model_api_key` and `user_model_name`. Start the servers with a sandbox provider:

```bash
gym env start \
    --resources-server bird_interact \
    --model-type vllm_model \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml
```

Rows route to an Environment Server by taskset, so write each example row as a materialized task, once for each
mode, and run them against the started servers:

```bash
python3 - <<'EOF'
import json
rows = [json.loads(line) for line in open("resources_servers/bird_interact/data/example.jsonl")]
with open("bird_interact_example_input.jsonl", "w") as f:
    for mode in ("c_interact", "a_interact"):
        for row in rows:
            task_id = {"taskset": f"bird_interact:{mode}_example", "task_id": row["instance_id"]}
            f.write(json.dumps({"task_id": task_id, "task_input": row}) + "\n")
EOF

gym eval run --no-serve \
    --resources-server bird_interact \
    --model-type vllm_model \
    --input bird_interact_example_input.jsonl \
    --output results/bird_interact_example.jsonl
```

## Licensing

Code: Apache 2.0. The example database and tasks: Apache 2.0. BIRD-Interact data: CC BY-SA 4.0.
