# Description

Agent harness for the [AppWorld](https://github.com/StonyBrookNLP/appworld)
interactive coding-agent benchmark. AppWorld is code-as-action: the policy model
writes Python against 457 APIs in a persistent IPython shell, one chunk per turn,
via a single `execute_ipython_code(code)` tool.

Two AppWorld properties make this a dedicated harness rather than `simple_agent`:

- **The task text is not in the dataset row.** AppWorld's tasks are part of its
  encrypted, redistribution-restricted portion, so rows carry only a task id and
  `/seed_session` returns the system prompt plus the supervisor/instruction turn
  as observations, which the harness prepends to the rollout.
- **Termination is decided by the environment.** An episode ends when the agent
  calls `apis.supervisor.complete_task()` inside the sandbox (or the interaction
  budget runs out) — a signal that arrives on the `/step` response, not as a stop
  from the model. A turn with no tool call is treated as giving up.

`/close` runs in a `finally`, so the leased AppWorld worker process is always
returned to the pool; it is also what triggers scoring on the resources server.

See `resources_servers/appworld/README.md` for architecture, setup, data and run
instructions.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0 with an added encrypted-redistribution requirement (StonyBrookNLP/appworld)

Dependencies
- nemo_gym: Apache 2.0
- tenacity: Apache 2.0
