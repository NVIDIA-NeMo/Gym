# LangGraph Agent

Reflection, orchestration, parallel thinking and ReWOO execute their complete graph
inside a task sandbox. The Agent Server manages sessions, installs the runtime,
transfers requests/results and supervises the graph process. Each graph calls the
Gym Model Server directly from the sandbox. Graph prompts and model-only behavior
are unchanged: these workflows do not execute environment tools or model-requested
function calls.

## Configure and run

Select a flavor config, a Model Server and a sandbox provider. For example, from the
Gym repository root, with the model endpoint configured:

```bash
gym env start \
    --config environments/langgraph_reflection_reasoning_gym/config.yaml \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
    --model-type vllm_model \
    ++use_absolute_ip=true

gym eval run --no-serve \
    --config environments/langgraph_reflection_reasoning_gym/config.yaml \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
    --model-type vllm_model \
    --agent langgraph_reflection_reasoning_gym_agent \
    --input environments/langgraph_reflection_reasoning_gym/data/example.jsonl \
    --output example_rollouts.jsonl --limit 1
```

Use the agent name declared by the selected environment config. The four flavors
retain their individual entrypoints (`reflection_agent.py`, `orchestrator_agent.py`,
`parallel_thinking_agent.py`, `rewoo_agent.py`); each is a thin adapter. The graph
implementations live in `workflows.py`, imported only by `sandbox_runner.py`.

The flavor configs under `configs/` set `resources_server: null`, so an Environment
Server can front them, like the other sandboxed harnesses. They are no longer unbound
swap sources: `--agent-type langgraph_agent/<flavor>_agent` does not replace a
benchmark's agent and leaves the benchmark running its own harness. To run a flavor on
another benchmark, compose it as the `environments/langgraph_*` pairings do: bind the
agent's `resources_server`, front it with `single_agent_turn_legacy` with
`resources_tool_transports: []` (the graphs refuse required tool grants), and load a
sandbox provider config.

Resources may supply `sandbox_access` during agent session setup. In that case the
agent connects to the existing task sandbox, honors its workdir/user/environment,
and leaves sandbox destruction to Resources. The provider must support reconnecting
from its descriptor. Otherwise `sandbox_provider` selects an agent-owned sandbox;
checked-in pairings use the named `sandbox` provider. Configure a provider block as
shown above, or override that name and `sandbox_config` for your deployment.

The default agent-owned image is `ubuntu:24.04`, with workdir `/workspace`. On a
cluster, select a supported provider/image and writable workdir. The task sandbox
must reach the Model Server and Python package/download endpoints. A model URL
reachable only on the Agent Server's loopback interface is usually insufficient.

## Runtime installation

The Agent Server never imports the graphs; its `requirements.txt` repeats the graph
pins only so the agent's tests can run them. At seed time the adapter bundles the
serving Gym source, graph implementations and runner, then installs them in a
session-local Python 3.13.14 environment inside the sandbox. `runtime-requirements.txt`
pins LangGraph and langchain-core, and every other dependency is constrained to the
version installed on the Agent Server. The source archive contains no server configuration,
model-provider keys or resource cookies. Only the Gym model URL and request/graph
settings are sent to the runner. Each session has its own install directory, avoiding
shared installation races; close removes it from a borrowed sandbox.

The sandbox supervisor bounds execution and confirms process cleanup before output
collection. Agent close drains the graph, disconnects a borrowed sandbox or stops
an owned one, then the Environment Server verifies and closes Resources. Identical
activation retries join/replay one graph; changed requests or paths return 409.
Required HTTP/MCP grants are refused before setup; optional grants are ignored. A graph
the supervisor stops at its deadline, or through `POST /rollouts/{rollout_id}/stop`,
returns an `incomplete` response with no output, which the Environment Server verifies
instead of retrying.

| Setting | Default | Purpose |
| --- | --- | --- |
| `sandbox_provider` | `null` in the config class; `sandbox` in recipes | Used only when Resources does not supply a sandbox |
| `sandbox_config` | Ubuntu 24.04, `/workspace` | Specification for an agent-owned sandbox |
| `sandbox_install_timeout_seconds` | 900 | Bounds runtime installation |
| `sandbox_runner_timeout_seconds` | 21600 | Graph activation ceiling; a shorter episode/request budget wins |
| `session_close_timeout_seconds` | 30 | Bounds supervised cleanup and connection release |

## Compatibility and evidence

**Capability level: Evaluation.** Both agent sessions and direct `/v1/responses`
requests execute inside sandboxes. Direct requests create and close a temporary
agent-owned sandbox and therefore require `sandbox_provider`; there is no host
execution fallback. Compatibility `/run` still requires `resources_server` and
preserves verifier extras and resource cookies. Existing flat-row pairings use
`single_agent_turn_legacy`, `resources_tool_transports: []` and one agent worker.

All model calls preserve rollout and training-capture URL prefixes. Output metadata
reports `harness_execution: sandbox` and the runner's hostname, PID and interpreter.
Usage is aggregated only when every model call reports it; otherwise it stays unknown.
Native invocation/turn evidence remains absent, so health can be `unobserved` and
strict evidence inspection is not expected to pass. Process identity is not a claim
of native observability or training qualification.

Checkpoint recovery remains `restart_only`. Non-monotonic trajectories, including
parallel reasoning and synthesis, are not directly compatible with NeMo RL's
monotonically increasing trajectory assumption. `max_output_tokens` remains a
per-model-call limit rather than a whole-graph budget. Moving the graph into the
sandbox does not change its prompts, tool capabilities or these training limits.
