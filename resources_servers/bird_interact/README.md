# BIRD-Interact Resources Server

Episode state for [BIRD-Interact](https://bird-interact.github.io/) ([paper](https://arxiv.org/abs/2510.05318)),
an interactive text-to-SQL benchmark. The request is ambiguous, and some knowledge-base entries are hidden. The agent
must ask a simulated user to resolve the ambiguity before it writes SQL.

One session holds one task:

- a copy of the task's SQLite database, with `preprocess_sql` applied;
- the two-stage user simulator, which calls the `user_model` server;
- the grader, the current phase, the SQL attempts and the reward;
- in a-Interact, the BIRD-COIN budget.

The `bird_interact` Environment Server (`environment_servers/bird_interact`) runs the protocol in both modes on
`BaseUserSimulationEnvironmentServer`.

## Modes and routes

| Mode | Who owns the loop | Routes |
|---|---|---|
| c-Interact | The Environment Server. Each agent reply is a question (`/ask`) or SQL (`/submit`). | `/episode_context`, `/ask`, `/submit`, `/verify` |
| a-Interact | The agent, in one turn. | Nine tools under `/a_interact/`: `ask`, `submit`, `execute`, `get_schema`, `get_all_column_meanings`, `get_column_meaning`, `get_all_external_knowledge_names`, `get_knowledge_definition`, `get_all_knowledge_definitions` |

Both modes use the same simulator and grader. The agent's tool grant covers only `/a_interact/`, so it cannot call
`/verify` or the c-Interact routes.

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

`data/example.jsonl` has three tasks on `data/example_databases/bookshop`, a synthetic database written for this
server. `bookshop_3` has a follow-up question.

The public BIRD-Interact releases (Mini-Interact, Lite, Full) withhold `sol_sql` and `test_cases`. The authors
send them on request. A seed for a row without `sol_sql` is refused with HTTP 422. Mini-Interact uses SQLite, so its
databases work as downloaded from [`birdsql/mini-interact`](https://huggingface.co/datasets/birdsql/mini-interact),
after the ground truth is merged into the rows.

## Not supported yet

- Rows with custom `test_cases` (BIRD-Interact's management tasks). The seed refuses them with HTTP 422.
- PostgreSQL (Lite and Full).
- Database changes that carry over from phase 1 to phase 2. Each submission runs on a scratch copy.

## Running

```bash
gym eval run \
    --resources-server bird_interact \
    --model-type vllm_model \
    --split example \
    --output results/bird_interact_example.jsonl
```

This runs the example tasks in both modes. `env.yaml` needs `policy_base_url`, `policy_api_key`,
`policy_model_name`, and the simulator's `user_model_base_url`, `user_model_api_key` and `user_model_name`.

## Licensing

Code: Apache 2.0. The example database and tasks: Apache 2.0. BIRD-Interact data: CC BY-SA 4.0.
