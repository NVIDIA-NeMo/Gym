# Harness agent

Runs a configured agent harness through a sandbox provider. LocalProvider runs the same
path without container isolation.

Supported `agent` values are `claude_code`, `cline`, `codex`, `hermes`, `kilocode`,
`openclaw`, `pi`, `prime`, and `terminus_2`. Ready-to-edit configurations are in
[`configs/`](configs/). Each uses the same interface:

```yaml
agent: hermes
agent_kwargs:
  model:
    model: ${policy_model_name}
    base_url: __SANDBOX_MODEL_URL__/v1
  timeout_seconds: 300
  max_turns: 30
```

Harness-specific options belong under `agent_kwargs.settings`. The selected config's
`setup_commands` install its CLI when it is not already available.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agent agnostic.
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | Working directory used by the agent and grader |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward lands in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |
Tasks with an external verifier (e.g. math) need none of these beyond an image.

## In-sandbox agent runtime

The server stages `nemo_gym` and the runner in each sandbox. `setup_commands` install
the selected harness CLI.

Because the harness inside the sandbox talks to a standard Gym model server, this agent
composes with future model-server capabilities (e.g. token-ID capture for training)
without changes to the harness or this server.
