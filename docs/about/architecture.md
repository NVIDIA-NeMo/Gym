(about-architecture)=
# Architecture

NeMo Gym runs a set of HTTP servers that coordinate rollout execution (see {doc}`concepts/key-terminology`). The main server roles are model servers for inference, resources servers for environment logic and verification, agent servers for orchestration, and a head server that publishes resolved configuration.

## TL;DR

- **Agent servers** orchestrate rollout execution by calling **model** and **resources** servers.
- **Resources servers** manage environment state and compute verification rewards.
- **Model servers** expose OpenAI-compatible inference endpoints.
- **The head server** publishes the resolved configuration for service discovery.

## High-Level Architecture

```{note}
Architecture diagram not yet available. This section describes the runtime interactions and endpoints defined in the server implementations.
```

## Components and Responsibilities

### Model servers (Responses API)

Model servers expose OpenAI-compatible inference endpoints for chat and responses:

- `POST /v1/chat/completions`
- `POST /v1/responses`

The base model server class defines these endpoints. Concrete model servers implement them (for example, the OpenAI-backed model server). Agents call these endpoints through the shared server client.

### Resources servers (environment + verification)

Resources servers expose environment lifecycle endpoints:

- `POST /seed_session` to initialize session state
- `POST /verify` to compute rewards from a rollout response

Individual resources servers can add domain-specific endpoints for tools or environment steps. For example:

- A resources server can register a catch-all tool route like `POST /{path}` for tool execution.
- Aviary-based resources servers add `POST /step` and `POST /close` for multi-step environments.

### Agent servers (rollout orchestration)

Agent servers expose two primary endpoints:

- `POST /v1/responses` for multi-step interaction
- `POST /run` for full rollout execution and verification

The base agent server class wires these routes, while each agent implementation defines how to call model and resources servers.

### Head server (configuration discovery)

The head server exposes:

- `GET /global_config_dict_yaml` to return the resolved global configuration
- `GET /server_instances` to list server instances started by the CLI

The shared server client fetches the resolved configuration from the head server and uses server names to resolve host/port for inter-server requests.

## Request Flow

### Rollout via `POST /run` (SimpleAgent)

The `SimpleAgent` implementation orchestrates a complete rollout and verification sequence:

1. Call the resources server `POST /seed_session` to initialize session state.
2. Call the agent `POST /v1/responses`. The agent calls the model server `POST /v1/responses` and issues tool calls to the resources server via `POST /{tool_name}`.
3. Call the resources server `POST /verify` and return the verified rollout response.

The rollout collection flow uses the agent `POST /run` endpoint and writes the returned metrics to JSONL output.

### Multi-step environments (Aviary example)

Some resources servers model environments with explicit step and close endpoints. Aviary-based resources servers accept `POST /step` for environment transitions and `POST /close` to release an environment instance.

## Session and State

All servers add session handling that assigns a session ID when one is not present. Agents propagate cookies between model and resources servers, which lets resources servers store per-session state. Several resources servers keep in-memory maps keyed by session ID (for example, counters or tool environments) to track environment state across steps.

## Configuration and Port Resolution

The global configuration dict configures server instances. During parsing, servers without explicit host/port values receive defaults (host defaults to `127.0.0.1`, and the system assigns ports from available ports). The head server uses port `11000` by default and publishes the resolved configuration used by the server client.

## Related

- {doc}`concepts/core-components` — Component overview and examples
- {doc}`concepts/configuration` — Configuration model and server references
- {doc}`../infrastructure/index` — Deployment guides
