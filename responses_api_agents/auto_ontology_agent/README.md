# Auto Ontology Agent

This agent runs [Auto Ontology](https://github.com/NVIDIA/auto-ontology)'s ReAct text-to-SQL agent in Gym: its
own loop, prompts and tools, on the policy model. `benchmarks/birdbench/auto_ontology_agent.yaml` pairs it with BIRD
dev. It also takes part in BIRD-Interact's c-Interact, which the `bird_interact` Environment Server runs (see below).

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
| `max_concurrent_episodes` | 8 | Episodes this server runs at once. Each session holds its sandbox and Postgres from seed to close, so the run's concurrency bounds the sandboxes |

## Known limits

- **Usage:** the Model Server's capture records usage (`+observability_enabled=true`). The reply carries no `usage`,
  and the close reports the `model_call_usage_unavailable` gap.
- **Retries:** a failed activation is not retried in the same sandbox; the Environment Server's retry starts a new
  episode.
- **Model failures:** when Auto Ontology's loop gives up on a model call, after its own retries, the reply is `failed`
  and carries the call's failure kind: `model_request_error` with the Model Server's HTTP status, or
  `model_transport_error` when the call got no reply. The Environment Server's outcome policy then grades, retries or
  fails the episode. An episode that ends with no query for another reason, such as its tool budget, is graded.
- **Request controls:** Auto Ontology's loop sets its own sampling, limits and instructions. An activation that sets
  another field, such as `temperature`, `max_output_tokens`, `tool_choice` or `instructions`, is refused with a 422.
- **BIRD-Interact:** c-Interact only; a-Interact's tools would stand in for Auto Ontology's own. Auto Ontology runs
  read-only queries, so it cannot solve a management task. The c-Interact trajectory holds each turn's input and
  answer; the model calls are in the Model Server's capture.
