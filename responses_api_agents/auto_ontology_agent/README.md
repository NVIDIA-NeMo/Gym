# Auto Ontology Agent

This agent runs [Auto Ontology](https://github.com/NVIDIA/auto-ontology)'s ReAct text-to-SQL agent in Gym: its
own loop, prompts and tools, on the policy model. `benchmarks/birdbench/auto_ontology_agent.yaml` pairs it with BIRD
dev.

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
| `sandbox_provider`, `sandbox_config` | required | The sandbox of each episode: a provider block of the global config, and the image |
| `catalog_python`, `catalog_workdir`, `catalog_start_command` | the catalog image's | In the image: the Python with Auto Ontology, the folder whose `.env` names the databases and embedding endpoint, and the command that starts the catalog |
| `environment` | none | Auto Ontology's settings in the sandbox, such as the embedding key (`DEFAULT_MODELS_API_KEY`) or `TEXT_TO_SQL_REACT_MAX_TOOL_CALLS`. They cannot change the model |
| `catalog_ports` | 20000 to 60000 | The ports an episode's catalog may take |
| `setup_timeout_seconds`, `episode_timeout_seconds`, `close_timeout_seconds` | 300, 1800, 60 | Limits on the seed, the episode and the close |
| `max_concurrent_episodes` | 8 | Episodes at once in this server, each with its own sandbox and Postgres |

## Known limits

- **Usage:** the Model Server's capture records usage (`+observability_enabled=true`). The reply carries no `usage`,
  and the close reports the `model_call_usage_unavailable` gap.
- **Retries:** a failed activation is not retried in the same sandbox; the Environment Server's retry starts a new
  episode.
