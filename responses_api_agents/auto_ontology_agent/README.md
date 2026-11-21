# Auto Ontology Agent

This agent runs [Auto Ontology](https://github.com/NVIDIA/auto-ontology)'s ReAct text-to-SQL agent in Gym: its
own loop, prompts and tools, on the policy model. `benchmarks/birdbench/auto_ontology_agent.yaml` pairs it with BIRD
dev. It also takes part in BIRD-Interact's c-Interact and a-Interact, which the `bird_interact` Environment Server runs
(see below).

## How an episode runs

```
Environment Server (single_agent_turn)
  seed     -> agent: start a sandbox from the catalog image, with episode_runner.py; start the catalog's Postgres
  activate -> agent: run Auto Ontology's run_episode in the sandbox
                sandbox -> the policy's Model Server (each model call, on the rollout's route)
                sandbox -> the catalog and the task's database (retrieve, run_sql, submit)
              <- the submitted query, in a ```sql block
  close    -> agent: stop the sandbox
  verify   -> bird_sql grades the query
```

- **The task** comes in the seed's agent context: `question`, `evidence` and `database`. The agent writes no prompt
  of its own and ignores the activation's input. Auto Ontology builds its prompt from the task.
- **The episode is Auto Ontology's.** `episode_runner.py` calls `run_episode` as Auto Ontology's `ReactSQLAgent`
  does for a SQL answer, with Auto Ontology's reasoning client pointed at the policy's Model Server.
- **One sandbox per episode**, from the image `benchmarks/birdbench/auto_ontology/build_image.sh` builds. That image
  holds the Auto Ontology version Gym pins, the databases and their catalog. The agent's process imports only Gym.

## BIRD-Interact's c-Interact

In c-Interact (`c_interact.py`) the `bird_interact` Environment Server opens up to four turns: phase 1, the follow-up
question as phase 2, and a debug turn after each phase that failed. The session is a multi-turn one:

```
Environment Server (bird_interact, c_interact)
  seed     -> agent: no task; a direct HTTP grant of ask_user and submit_sql; start the sandbox and its catalog
  turn N   -> agent: run Auto Ontology's run_episode in the sandbox, in the turn's own directory
                sandbox -> the policy's Model Server (each model call)
                sandbox -> the catalog and the release's databases (retrieve, run_sql, submit)
                sandbox -> the Environment Server's ask_user (each question to the simulated user)
                sandbox -> the Environment Server's submit_sql (the query Auto Ontology accepts)
              <- the submitted query and the server's reply
  close    -> agent: stop the sandbox; return each turn's input, answer and duration
```

- **The task** comes in the ADK's texts. The first request's system message names the database, the visible
  knowledge and how many questions the agent may ask; each turn's user message is the request, the debug message or
  the follow-up question. The agent reads them and refuses a request that does not hold them.
- **Each turn is one Auto Ontology episode**, with Auto Ontology's prompts and tools, `ask_user` added through the hook
  that Auto Ontology's `run_episode` has for extra tools, and a section on asking in its system prompt. The evidence
  carries the visible knowledge, the user's earlier answers and the earlier queries, with the server's reply to the
  last. A turn that ends without an accepted query submits nothing.
- **The sandbox calls the Environment Server's tools**, so the grant's URL must be reachable from the sandbox, as the
  Model Server's must. A call the server fails or refuses replies with its error, which the model reads as the tool's
  result; the episode goes on.
- **An episode that raises** writes its error to the result file, and the activation fails with that error.

### The interactive pipeline

With `pipeline: interactive`, the agent runs Auto Ontology's interactive pipeline
(`auto_ontology.retrieval.interactive`) in place of its ReAct agent: the pipeline auto-ontology-eval runs on
BIRD-Interact, with checks that decide whether to ask, each answer merged into the question, and the follow-up
rewritten to stand alone. It writes its SQL with Auto Ontology's text-to-SQL, which is the ReAct agent from Auto
Ontology 1.3.0. The agent does what auto-ontology-eval's adapter (`ontology_sql_eval/bird_interact/server.py` at
e727e4d6) does around it, in c-Interact and in a-Interact:

- **The session** is created as the adapter creates it: the schema and visible knowledge, the question budget,
  `skip_evidence_refinement` and `numeric_type_alignment`.
- **A question** (`AskUserAction`) goes to the server's question tool, and the answer, without the server's note on
  questions left, goes back to the pipeline. **The query** (`SubmitSQLAction`) goes to the server's submit tool,
  without the `<database>.` before each table, and the reply goes back as the result.
- **The knowledge's links:** with `knowledge_dir`, the pipeline also gets each visible entry's `children_knowledge`
  (up to 5, among the visible entries), from `<knowledge_dir>/<db>/<db>_kb.jsonl`, as the adapter reads them. The
  BIRD-Interact ADK does not show agents that field; leave `knowledge_dir` unset to give the pipeline only what the
  ADK shows.
- **Settings** the pipeline reads (`INTERACTIVE`) go in `environment`. The BIRD-Interact configs copy the flags
  auto-ontology-eval's runs set; Auto Ontology 1.3.0 reads only `INTERACTIVE` of them.
- **The models:** Auto Ontology's reasoning role runs on the policy, and its non-reasoning role on
  `non_reasoning_model_server` when it is set (auto-ontology-eval runs Claude Sonnet 5.5 there beside Claude Opus 5.5),
  else on the policy too. Both are Gym Model Servers, so Gym captures every call on the rollout's route.

**c-Interact.** Each turn steps the pipeline until it submits, at most the question budget plus 5 steps. The session is
saved in the sandbox's session folder after each turn and loaded at the next, with its connections made again.

**a-Interact** runs as one activation, as auto-ontology-eval runs it. The agent first buys the schema, the column
meanings and the knowledge base through the server's tools (3 BIRD-COIN; the column meanings are paid for but not read,
as the catalog can hold them). Then one call in the sandbox runs phase 1, its retries and the follow-up within the
budget, by auto-ontology-eval's rules (`ontology_sql_eval/bird_interact/budget.py`): phase 1's queries and questions
leave 4 coins; the pipeline asks while the budget still pays for a submission; a submission is never refused, and one
that spends the budget ends the task; a retry that repeats a failed query ends the phase; the follow-up gets
auto-ontology-eval's instructions on extending phase 1's query. Each query Auto Ontology runs on the database goes to
the server's `execute`, which charges it, or refuses it once the budget is spent, and is then read from the sandbox's
own copy of the database: `execute` shows at most 20 rows as text, and Auto Ontology needs the whole result, as
auto-ontology-eval's connector returns it. The ReAct agent is told the balance, in auto-ontology-eval's words.

A seed for the interactive pipeline must grant the server's tools over direct HTTP and carry no task context.

## The sandbox's network

The episode calls the policy's Model Server from the sandbox. The config runs local Docker sandboxes on the host
network, since a host firewall often closes Gym's address to Docker's bridge. Sandboxes then share loopback, so the
agent starts each episode's catalog on its own port, at random from `catalog_ports`, and tries another port when one is
taken. On a remote sandbox provider, give the Model Server an address the sandbox can reach (`++use_absolute_ip=true`).

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `model_server` | required | The policy's Model Server |
| `model` | `openai/policy` | The name Auto Ontology sends. The `openai/` prefix selects its OpenAI-compatible client; the Model Server uses its own model name |
| `non_reasoning_model_server`, `non_reasoning_model` | none, `openai/non-reasoning` | The Model Server of Auto Ontology's non-reasoning role, and the name it sends; none runs that role on the policy |
| `sandbox_provider`, `sandbox_config` | required | The sandbox of each episode: a provider block of the global config, and the image |
| `catalog_python`, `catalog_workdir`, `catalog_start_command` | the catalog image's | In the image: the Python with Auto Ontology, the folder whose `.env` names the databases and embedding endpoint, and the command that starts the catalog |
| `environment` | none | Auto Ontology's settings in the sandbox, such as the embedding key (`DEFAULT_MODELS_API_KEY`) or `TEXT_TO_SQL_REACT_MAX_TOOL_CALLS`. They cannot change the model |
| `catalog_ports` | 20000 to 60000 | The ports an episode's catalog may take |
| `setup_timeout_seconds`, `episode_timeout_seconds`, `close_timeout_seconds` | 300, 1800, 60 | Limits on the seed, the episode and the close |
| `max_concurrent_episodes` | 8 | Episodes at once in this server, each with its own sandbox and Postgres |
| `pipeline` | `react` | Auto Ontology's ReAct agent, or its interactive pipeline (BIRD-Interact only, c-Interact or a-Interact) |
| `knowledge_dir` | none | The interactive pipeline's knowledge links: a folder with `<db>/<db>_kb.jsonl` |

## Known limits

- **Usage:** the Model Server's capture records usage (`+observability_enabled=true`). The reply carries no `usage`,
  and the close reports the `model_call_usage_unavailable` gap.
- **Retries:** a failed activation is not retried in the same sandbox; the Environment Server's retry starts a new
  episode.
- **BIRD-Interact:** the ReAct agent runs c-Interact only; a-Interact runs with the interactive pipeline, which uses
  the server's tools for questions, queries and submissions and Auto Ontology's catalog for the rest. Auto Ontology
  runs read-only queries, so it cannot solve a management task. The trajectory holds each activation's input and
  answer; the model calls are in the Model Server's capture.
