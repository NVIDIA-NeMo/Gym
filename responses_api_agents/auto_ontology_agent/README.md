# Auto Ontology Agent

Runs [Auto Ontology](https://github.com/NVIDIA/auto-ontology)'s ReAct text-to-SQL agent as a Gym Agent Server. Its first
pairing is BIRD dev (`benchmarks/birdbench/auto_ontology.yaml`), graded by the `bird_sql` Resources Server.

| | |
|---|---|
| Harness | Auto Ontology's `main` at `a630ebd`, plus the commit of NVIDIA/auto-ontology#543 that accepts the `openai` version Gym pins (`ffrujeri/auto-ontology@e774af9f`). Move to `NVIDIA/auto-ontology` once #543 merges. |
| Execution | In process: Auto Ontology is a Python library, so it runs in the Agent Server's process, with no sandbox |
| Capability level | Observability: each model call carries its rollout and its invocation. Not yet checked on saved rollouts |
| Pairing | BIRD dev, through `single_agent_turn` and `bird_sql` (`benchmarks/birdbench/auto_ontology.yaml`) |
| Tool grants | None: Auto Ontology calls its own tools |
| Dependencies | `pyproject.toml` and `uv.lock`. Gym's root uv policy excludes `sqlalchemy` and `alembic` from server venvs. `scripts/component-lock-policy.json` keeps them for this agent, because Auto Ontology's catalog code imports them. |

## How it works

Auto Ontology answers a question with one ReAct episode (`auto_ontology.retrieval.text_to_sql.react.loop.run_episode`):
the model searches Auto Ontology's catalog (`retrieve`), runs SQL on the source database (`run_sql`), and submits a query
(`submit`), within Auto Ontology's budget of tool calls. The agent calls `run_episode` as Auto Ontology's own ReAct
step does, with the settings auto-ontology-eval uses for a graded benchmark: SQL answers only, the question as asked,
and the evidence apart.

- **The task.** The Environment Server seeds one agent session for each task, and the row's `agent_context` gives the
  agent the question, its evidence and its database (`AutoOntologyTask`). The agent refuses a seed without them, with
  other keys, with a task sandbox, with a required tool grant or with continuation. `prepare.py
  +prepare_script_args.agent_context=true` writes these rows for BIRD dev.
- **The activation.** The agent runs the episode in a thread and answers with the submitted query in a fenced `sql`
  block, which `bird_sql` grades, or with a sentence naming the end reason when the episode submitted none. The
  activation's own input, the benchmark's prompt, is not used: Auto Ontology writes its prompts from the task. The
  response's metadata gives Auto Ontology's end reason and its tool-call count. An activation that is sent again gets
  the same reply, with no second episode.
- **The output text** is not Auto Ontology's chat reply: its pipeline formats a reply after the ReAct step, with
  another model call, and the agent skips that step, because the query is what is graded.

## Model calls

The episode's model is Auto Ontology's reasoning client, built by its own `get_llm_client`, so it keeps Auto Ontology's
settings (`max_tokens`, timeout, no temperature). The agent gives it an HTTP transport (`model_calls.py`) that sends each
call through Gym's `ServerClient`:

- The Environment Server activates the session on its rollout's route: `/ng-rollout/<capture key>`, with the
  training-capture segment when training capture is on. Each model call of the episode takes the same route, so the
  Model Server captures the call for its rollout.
- Each call sends the invocation ID (`root`) as `x-session-id`. Gym uses it to give the captured call to its
  invocation.
- The response carries the usage of the episode's model calls, when every call reported its usage.

The activation keeps the route in a context variable, which `asyncio.to_thread` copies into the episode's thread. A call
that still loses the context reaches its Model Server without a rollout. The transport counts it, and each session that
was open at the time declares `model_call_ownership_unavailable` at close.

## Setup

Auto Ontology retrieves from its catalog and runs SQL on the source databases, so both must exist before a run.

1. **Catalog.** A PostgreSQL database with pgvector, such as the `pgvector/pgvector:pg17` image that Auto Ontology's
   `docker-compose.yml` uses. Apply Auto Ontology's migrations from the pinned revision: `alembic upgrade head` in a
   checkout of it, with the `POSTGRES_*` variables set.
2. **Source databases.** BIRD dev's SQLite files, which `gym eval prepare` downloads for `bird_sql`.
3. **Ingest and compile** each database into the catalog, as auto-ontology-eval does, with the same connection strings
   as `source_database_url` gives: the catalog names each database by its ingest's connection. Ingest and the agent must
   use the same embedding model.
4. **Settings** in `env.yaml`:

   ```yaml
   auto_ontology_embedding_base_url: <embedding endpoint, OpenAI-compatible>
   auto_ontology_embedding_api_key: <its key>
   auto_ontology_embedding_model: <the model the catalog was ingested with>
   auto_ontology_catalog_password: <catalog password>
   auto_ontology_source_database_url: sqlite:///<dev_databases directory>/{database}/{database}.sqlite
   ```

   `configs/auto_ontology_agent.yaml` sets the catalog's host, port, user and database; override them there.

## Run

```bash
gym eval prepare --config benchmarks/birdbench/auto_ontology.yaml
gym eval run --config benchmarks/birdbench/auto_ontology.yaml --model-type vllm_model --split benchmark \
  --limit 200 --output results/birdbench_auto_ontology.jsonl
```

`--limit 200` runs BIRD dev's first 200 tasks, the subset Auto Ontology's ReAct results report. The model name the
agent sends, `openai/${policy_model_name}`, needs an `openai/`, `azure/` or `aws/` prefix, which selects Auto Ontology's
OpenAI-compatible client, the only one whose calls go through Gym. Gym's Model Servers send their own model name.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `model_server`, `model` | `policy_model`, `openai/${policy_model_name}` | The model the episode runs on: the policy |
| `embedding_*` | from `env.yaml` | Auto Ontology's embedding endpoint, which it calls directly |
| `catalog` | `localhost:5432`, `postgres`, `auto_ontology` | The catalog database |
| `source_database_url` | from `env.yaml` | The source databases, with a `{database}` field |
| `feature_flags` | none | Environment variables that Auto Ontology reads at import, such as `TEXT_TO_SQL_REACT_MAX_TOOL_CALLS` (default 24), `TEXT_TO_SQL_REACT_SUBMIT_TOOL` (default on) and `TEXT_TO_SQL_REACT_GREP` (default off) |
| `numeric_type_alignment` | `false` | As auto-ontology-eval leaves it |
| `max_concurrent_episodes` | `8` | Episodes that run at once, in threads |

## Limits

- **Embedding calls do not go through Gym.** Auto Ontology calls its embedding endpoint directly.
- **One Auto Ontology for each process.** Auto Ontology reads its settings and feature flags at import, so one Agent
  Server process serves one set of them.
- **A running episode finishes in its thread.** An episode that its caller abandons still runs to its end.
- **No training yet.** The training-capture route reaches every model call of the episode, which is one conversation;
  training on it is not tried yet.
