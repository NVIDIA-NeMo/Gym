# BIRD-Interact Environment Server

[BIRD-Interact](https://bird-interact.github.io/) ([paper](https://arxiv.org/abs/2510.05318)) is an interactive
text-to-SQL benchmark. The request is ambiguous, and some knowledge-base entries are hidden. The agent must ask a
simulated user to resolve the ambiguity before it writes SQL.

This Environment Server runs both of its modes with no Resources Server. It is built on
`BaseUserSimulationEnvironmentServer` (`nemo_gym/base_user_simulation_environment_server.py`), which runs the turns
and serves the episode's tools through `BaseEnvironmentServerWithTools`. Each episode holds one task:

- a sandbox with the task's SQLite database, with `preprocess_sql` applied (see [Database sandbox](#database-sandbox));
- the two-stage user simulator, which calls the `user_model` server;
- the grader, the current phase, the SQL attempts and the reward;
- in a-Interact, the BIRD-COIN budget and the nine tools.

## Modes

| `mode` | Turns | Agent's tools | `next_user_turn` |
|---|---|---|---|
| `c_interact` | Up to `max_turns`. The agent must resume its session across turns. | None | The reply is the text of the turn's last message. A reply with a ```` ```sql ```` or ```` ```sqlite ```` block is a submission; any other reply is a question. The next user message is the simulator's answer, the debug prompt or the follow-up question. |
| `a_interact` | One (`max_turns: 1`). | Nine tools that the server serves for the episode, over each of `tool_transports`: `ask`, `submit`, `execute`, `get_schema`, `get_all_column_meanings`, `get_column_meaning`, `get_all_external_knowledge_names`, `get_knowledge_definition`, `get_all_knowledge_definitions` | `Stop("agent_finished")` |

The first turn's request has the instructions and the user's ambiguous request. In c-Interact it also has the schema,
the column meanings and the visible knowledge base. In a-Interact it leaves those out, because the tools charge for
them. Over `direct_http` it declares the nine function tools; over `mcp` alone it declares none, because the agent
lists them from the server and Codex refuses a request that declares tools.

**Tools.** The agent calls the tools with a signed episode token that its seed grants. Over `direct_http` it calls
`POST /<tool name>` on this server; over `mcp`, `/mcp` lists only its episode's tools. The token works only on the
tool routes, for this episode, until the episode ends. With several workers, each call reaches the worker that runs
its episode. The tools have the same names, descriptions and arguments over both transports.

**c-Interact rules.** The agent may ask as many questions as the task has critical and knowledge ambiguities, plus
`question_patience` (default 3), before its first submission. A submission that fails gets one more attempt, which
must be SQL. A task with a `follow_up` asks it as phase 2 after phase 1 passes; phase 2 has the same two attempts.

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

**Result.** The base fields (`reward`, `turns`, `stop_reason`, `tool_calls`, agent observations), plus `phases`,
`questions_asked`, `coins_spent` and `finished`. Stop reasons: `solved`, `failed` (the second attempt failed),
`no_sql_submitted` (a question after the question budget, after a submission, or in phase 2), `agent_finished`
(a-Interact), and the base class's `max_turns` and `interaction_timeout`.

## User simulator

Stage 1 classifies the agent's question as an annotated ambiguity (`labeled`), a detail that the ground truth
settles but that is not annotated (`unlabeled`), or a question to refuse (`unanswerable`). Stage 2 writes the reply
from that action only. A refused question gets a fixed reply without a model call. A term in the question that
links to a hidden knowledge entry, through the same SQL fragment, lets the reply use that entry.

The prompts are written for Gym, not copied from the BIRD repository. Parity runs against the published scores need
the reference prompts and the reference simulator model (`gemini-2.0-flash-001`).

If the user model fails during an a-Interact `ask`, the result masks the sample with
`failure_kind: bird_interact:user_simulator_failed`. In c-Interact the failure fails the episode, which is retried.

## Database sandbox

The episode starts a sandbox and stops it when the episode ends, on every path; at shutdown the server finishes
the cleanups of episodes still running, and `sandbox_config.ttl_s` lets the provider reclaim a sandbox that a
crashed server left. The server uploads `grading.py` and the database source and runs
`python3 grading.py <setup|execute|grade> <request.json>` there, so the image needs only `python3` (default
`python:3.12-slim`). The agent's SQL and the grading run in the sandbox, never in the server process, and the agent
never gets the sandbox itself. `grading.py` deletes each request file once it has read it, so the solution SQL in a
grade request does not stay in the sandbox.

`sandbox_provider: sandbox` names the provider block that a provider config binds, such as
`nemo_gym/sandbox/providers/docker/configs/docker.yaml`. The local provider
(`nemo_gym/sandbox/providers/local/configs/local.yaml`) runs the commands on the host, which the tests use.

A command that the sandbox does not run, such as a provider error, a timeout or a crash of `grading.py`, is an
infrastructure failure, not the agent's mistake. It runs again on a fresh sandbox up to
`inconclusive_verification_retries` times (default 1). If it still fails, c-Interact fails the episode so that it is
retried, and a-Interact masks the sample with `failure_kind: bird_interact:sandbox_failed`.

## Grading

`grading.py` is an independent implementation of BIRD-Interact's default test case. It removes comments, `DISTINCT`
and `ROUND(...)` from both queries, runs each on a scratch copy of the episode's database, rounds floats to two
places, and compares rows in order when `conditions.order` is set, or as sets otherwise. An empty result fails.

## Harnesses

What each harness's Gym adapter accepts today, read from its session seed checks. `simple_agent`, in both modes,
and Codex, in a-Interact over `mcp`, have run BIRD with a model; the other rows are not results from runs.

| Harness | c-Interact | a-Interact |
|---|---|---|
| `simple_agent` | Yes | Yes, over `direct_http` |
| Codex, Pi, Hermes | No: the adapter has no multi-turn sessions | Yes, over `mcp`, in the harness's own sandbox |
| Stirrup | No: no multi-turn sessions | Yes, over `direct_http`, in its own sandbox. `code_exec: false` leaves out its shell. |
| OpenCode | Yes, in its own sandbox. `opencode_config.tools` turns off its built-in tools. | No: it refuses required tool grants |
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
<db>/<db>.sqlite (or <db>.sql)
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
after the ground truth is merged into the rows.

## Not supported yet

- Rows with custom `test_cases` (BIRD-Interact's management tasks). The episode fails at seed, for good.
- PostgreSQL (Lite and Full).
- Database changes that carry over from phase 1 to phase 2. Each submission runs on a scratch copy.
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

Code: Apache 2.0. The example database and tasks: Apache 2.0. BIRD-Interact data: CC BY-SA 4.0.
