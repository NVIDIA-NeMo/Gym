# Harbor harness agent

This agent runs any [Harbor](https://www.harborframework.com/) agent in a task sandbox that a Resources Server owns, such
as [`harbor_tasks`](../../resources_servers/harbor_tasks/README.md). It is session-only: an Environment Server seeds it
with `SandboxAccess`, activates it once, and closes it before verification.

1. **Seed** (`/v1/agent_sessions`): connect to the borrowed sandbox, wrap it as a Harbor environment
   (`nemo_gym.sandbox.adapters.harbor`), create the configured Harbor agent, and run its `setup()`, which installs
   installed-agent harnesses inside the sandbox.
2. **Activate** (`/ng-rollout/{capture_key}/v1/responses`): run the agent on the instruction from the request's input
   messages, within the task's agent timeout. Download its logs, let it convert them to ATIF, and return the ATIF
   trajectory as Gym Responses output.
3. **Close**: cancel an unfinished activation and disconnect without stopping the sandbox; the Resources Server
   verifies and stops it.

A timed-out agent, or one that exits non-zero, returns a response with `status: incomplete` and the reason in
`metadata.harbor_agent_exit`, and the episode still verifies, as in a Harbor trial.

## Choosing the Harbor agent

The configs leave `resources_server` unset, so any benchmark that pairs a Resources Server with this agent can swap
the Harbor agent at start-up with `--agent-type`:

| `--agent-type` | Harbor agent | Where it runs |
| --- | --- | --- |
| `harbor_harness_agent` | Terminus-2 | this server process; commands run in the sandbox |
| `harbor_harness_agent/opencode` | OpenCode | inside the sandbox, which must reach the model server |

```bash
gym env start --benchmark harbor/hello_world --agent-type harbor_harness_agent/opencode \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml --model-type vllm_model
```

Add a flavor for another Harbor agent as one more file in `configs/`. A swap keeps the benchmark's Resources Server,
model server and datasets; everything else, `logs_dir` included, comes from the flavor.

## Configuration

`harbor_agent` follows `harbor.models.trial.config:AgentConfig`: `name` (for example `terminus-2`, `opencode`,
`claude-code`) or `import_path`, `model_name`, `kwargs`, `env`, `override_timeout_sec`, `max_timeout_sec`,
`override_setup_timeout_sec`. Sampling settings such as temperature go in `kwargs`, because Harbor agents take them when
they are created at seed, before the activation request arrives.

Model calls go to the Gym model server named by `model_server`, under the rollout-prefixed base URL, so the model server
captures token ids for training (`token_id_capture: true`). Terminus-2 calls the model from this process and takes the
base URL as a constructor kwarg (`model_base_url_kwarg: api_base`). Installed agents run inside the sandbox: set
`model_base_url_kwarg: null` and list their base-URL environment variables in `model_base_url_env`, and make sure the
model server is reachable from the sandbox. Leave `model_server` unset to let the agent call its own provider directly.

The task's agent timeout and user arrive in the request's `metadata` (`harbor_agent_timeout_sec`,
`harbor_agent_user`); `harbor_agent.override_timeout_sec`, `max_timeout_sec` and `agent_timeout_multiplier` adjust the
timeout as in `harbor run`. Each episode's agent logs and `trajectory.json` land under `logs_dir/<capture_key>/`.

## ATIF conversion

The source `trajectory.json` stays the lossless, authoritative artifact. The response carries the closest Responses
representation, and every lossy conversion is reported as an `atif_conversion_lossy` observation gap:

- Scalar user, system, and assistant messages, function calls, and scalar tool outputs convert directly, in order.
- User and system multimodal content converts to `input_text` and `input_image` parts when images use portable URLs.
- Assistant multimodal content has no Responses output representation, so its ATIF content array is serialized as JSON.
- Multimodal tool outputs convert to content parts for HTTP, HTTPS, or data URLs; local image paths are serialized as JSON.
- Missing tool-call IDs get deterministic synthetic IDs, and missing tool output becomes empty text.
- Training metadata is emitted only when prompt IDs, non-empty completion IDs, and aligned log probabilities describe one
  attributable LLM output.

## Licensing information

Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- harbor: Apache 2.0
