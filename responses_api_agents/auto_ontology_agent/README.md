# Auto Ontology Agent

Runs [Auto Ontology](https://github.com/NVIDIA/auto-ontology), NVIDIA's text-to-SQL agent, as a Gym Agent Server on the
`bird_interact` Environment Server, in BIRD-Interact's c-Interact mode.

| | |
|---|---|
| Harness | Auto Ontology at `dd78d0689ef9a8e062af90a294fa903cb6643015`, the revision auto-ontology-eval pins at `8657bd6` |
| Execution | In process: Auto Ontology is a Python library, so it runs in the Agent Server's process, with no sandbox |
| Capability level | Evaluation |
| Pairing | `bird_interact`, c-Interact (`benchmarks/bird_interact/auto_ontology.yaml`) |
| Tool grants | One direct HTTP grant: the episode's `ask_user` and `submit_sql` |
| Dependencies | `pyproject.toml` and `uv.lock`. Gym's root uv policy excludes `sqlalchemy` and `alembic` from server venvs. `scripts/component-lock-policy.json` keeps them for this agent, because Auto Ontology's catalog code imports them. |

## How it works

The `bird_interact` Environment Server seeds one agent session for each episode, with continuation, and activates it
once for each turn. Auto Ontology's interactive mode advances a session one action at a time: ask the user, or submit
SQL. In each turn, the agent steps Auto Ontology until it submits, and runs each action as an episode tool.

| Auto Ontology needs | The agent reads it from |
|---|---|
| Database, schema, knowledge, question budget | The first activation's system message: the BIRD-Interact ADK's c-Interact instruction |
| The question | The first activation's user message: the ADK's phase 1 message |
| The next turn's kind and text | The next activation's message: the ADK's follow-up, or debug message with the execution error |
| An answer from the user | The `ask_user` reply, without the server's note on the clarifications left |
| Whether phase 1 passed | The `submit_sql` reply's first line, the grader's message |

The agent reads the ADK's texts exactly (`c_interact.py`). A text it does not know fails the activation with HTTP 422,
so a protocol change cannot pass unnoticed. Before it submits, the agent drops the database name that Auto Ontology
puts before each table, because the grader runs the SQL in another database (`<database>_template`). auto-ontology-eval
does the same.

## Setup

Auto Ontology retrieves from its catalog and probes values in the source databases, so both must exist before a run.

1. **Catalog.** A PostgreSQL database with pgvector, such as the `pgvector/pgvector:pg17` image that Auto Ontology's
   `docker-compose.yml` uses. Apply Auto Ontology's migrations from the pinned revision: `alembic upgrade head` in a
   checkout of it, with the `POSTGRES_*` variables set.
2. **Source databases.** The BIRD-Interact databases, for example a container of the recipe's image:
   `docker run -d -p 127.0.0.1:55433:5432 benchmarks/bird_interact:lite`.
3. **Ingest and compile** each database into the catalog, as auto-ontology-eval does, with its
   `ontology_sql_eval.ingestion.ingest` and `ontology_sql_eval.ingestion.semantic` modules. Ingest and the agent must use
   the same embedding model.
4. **Settings** in `env.yaml`:

   ```yaml
   auto_ontology_embedding_base_url: <embedding endpoint, OpenAI-compatible>
   auto_ontology_embedding_api_key: <its key>
   auto_ontology_embedding_model: <the model the catalog was ingested with>
   auto_ontology_catalog_password: <catalog password>
   auto_ontology_source_database_url: postgresql://root:<password>@127.0.0.1:55433/{database}
   ```

   `configs/auto_ontology_agent.yaml` sets the catalog's host, port, user and database; override them there.

## Run

```bash
gym eval prepare --config benchmarks/bird_interact/auto_ontology.yaml \
  +prepare_script_args.ground_truth_jsonl=<ground truth file>
gym eval run --config benchmarks/bird_interact/auto_ontology.yaml --model-type vllm_model --split benchmark \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml --output results/bird_interact_lite_c_ao.jsonl
```

Auto Ontology's reasoning and fast models go through the configured Gym Model Server (`policy_model` by default). The
model name it sends selects its client: an `openai/`, `azure/` or `aws/` prefix selects its OpenAI-compatible client.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `reasoning_model_server`, `reasoning_model` | `policy_model`, `${policy_model_name}` | Auto Ontology's reasoning model |
| `fast_model_server`, `fast_model` | the reasoning model | Auto Ontology's fast model |
| `embedding_*` | from `env.yaml` | Auto Ontology's embedding endpoint, which it calls directly |
| `catalog` | `localhost:5432`, `postgres`, `auto_ontology` | The catalog database |
| `source_database_url` | from `env.yaml` | The source databases, with a `{database}` field |
| `feature_flags` | auto-ontology-eval's BIRD-Interact flags | Environment variables that Auto Ontology reads at import |
| `knowledge_children_dir` | `null` | See below |
| `skip_evidence_refinement`, `numeric_type_alignment` | `true`, `true` | As auto-ontology-eval sets them |
| `step_margin` | `5` | Steps a turn may take beyond the question budget before the agent ends it without a submission |
| `max_concurrent_steps` | `8` | Auto Ontology steps that run at once, in threads |

**Access beyond the protocol.** auto-ontology-eval gives Auto Ontology each knowledge entry's `children_knowledge`,
which the ADK hides from agents. Set `knowledge_children_dir` to a directory of `<database>/<database>_kb.jsonl` files to
do the same, for example to compare with auto-ontology-eval's runs. Leave it `null` for scores that compare with other
agents. Auto Ontology also probes the source databases and its catalog directly, with no cost in either mode.

## Limits

- **c-Interact only.** A first activation without `ask_user` and `submit_sql`, such as a-Interact's, gets HTTP 422.
  a-Interact needs a loop in Auto Ontology that chooses the lookup tools.
- **Query tasks.** Auto Ontology writes one `SELECT`, so it fails management tasks until it has a path for DDL and DML.
- **Model calls are not attributed to rollouts.** Auto Ontology builds its model clients once for the process, so its
  calls reach the Model Server without the rollout prefix. The close bundle declares `model_call_ownership_unavailable`
  and `model_call_usage_unavailable`, and responses carry no usage. Token capture for training needs each of Auto
  Ontology's model calls to carry its rollout.
- **One Auto Ontology for each process.** Auto Ontology reads its settings and feature flags at import, so one Agent
  Server process serves one set of them.
- **A running step finishes in its thread.** Closing a session stops it between steps, not during one.
