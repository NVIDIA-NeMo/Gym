# NOOA Agent

The NOOA agent adapter provides typed configuration for constructing NOOA agents and invoking them through
asynchronous Python adapters that receive validated NeMo Gym Responses requests.

## Default agent and installation

The reusable `configs/nooa_agent.yaml` configuration runs `nooa_bench.bench_agent:BenchAgent` with
`responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent`. The adapter passes the task prompt to
BenchAgent and preserves its structured result. Set `nooa.agent_class` and `nooa.invocation_adapter` together
when selecting another agent.

Install the agent's requirements from its directory:

```bash
cd responses_api_agents/nooa_agent
uv sync --locked
```

The requirements pin all three NOOA packages: `nooa`, `nooa-cli`, and `nooa-bench`. The CLI supplies executor
support, and the benchmark package supplies the default BenchAgent. Gym's server installation uses this same
requirements file.

## Embedded CodeAct scope

Embedded CodeAct persists for the duration of one invocation. `execute_python` is not computation-only: it can
read and modify files, start processes, use the network, and change other node state. Embedded mode provides no
sandbox protection. Only injected resource methods are Gym/verifier-observable and authoritative; direct CodeAct
side effects belong to NOOA and are not reliably visible to verification or replay. The trajectory distinguishes
the outer `execute_python` call from nested resource calls.

## Atom

Atom (`nooa_atom.gym:AtomHarness`, installed from the same NOOA revision) runs through this adapter in place of the
bench agent. Three settings on the agent instance select it: the agent class, the invocation adapter
(`atom_adapter.invoke_atom`, which hands the latest user message to Atom's session and returns its `TaskResult`), and
Atom's root turn method, `handle_batch` (one agent, unattended) or `handle_orchestrate` (the root may delegate to child
sessions). `configs/nooa_atom_agent.yaml` sets the three on an agent instance, and `benchmarks/<bench>/atom.yaml`
(Terminal-Bench 2.1, SWE-bench Pro, GDPval) applies them to that benchmark's NOOA binding, so a run is one config:

```bash
gym eval run --config benchmarks/terminal_bench_2_1/atom.yaml \
  ++agent.responses_api_agents.nooa_agent.context_window=1048576 \
  ++agent.responses_api_agents.nooa_agent.max_output_tokens=131072 \
  ++agent.responses_api_agents.nooa_agent.temperature=1.0 \
  ++agent.responses_api_agents.nooa_agent.top_p=0.95 \
  ++agent.responses_api_agents.nooa_agent.reasoning_effort=none \
  '++agent.responses_api_agents.nooa_agent.media_inputs=[]'
```

The six `++agent...` overrides describe the policy model (see [Policy model settings](#policy-model-settings)); the
values shown are GLM 5.3's. The same three Atom settings as command-line overrides, on a binding whose agent instance
is named `agent`:

```bash
gym eval run --config benchmarks/terminal_bench_2_1/nooa.yaml \
  ++agent.responses_api_agents.nooa_agent.nooa.agent_class=nooa_atom.gym:AtomHarness \
  ++agent.responses_api_agents.nooa_agent.nooa.invocation_adapter=responses_api_agents.nooa_agent.atom_adapter:invoke_atom \
  ++agent.responses_api_agents.nooa_agent.nooa.init_kwargs.turn_method=handle_batch \
  <the six policy model settings as above>
```

Atom's session wraps a strategy failure in `TurnFailedError`; the adapter re-raises the wrapped error so the runner's
termination policy sees the same CodeAct stops and model rejections it sees from any other NOOA agent. Gym resource
tools are bound on the harness class, not on Atom's session agent.

## Policy model settings

The agent server declares the policy model as NOOA must know it, and refuses to start without every value: a run
evaluates a model it has described, never one the adapter guesses at. The shipped configs leave these keys mandatory
(`???`), so a benchmark config, an Eval Factory fragment or `++agent.responses_api_agents.nooa_agent.<key>=<value>`
overrides must set them before the config resolves. Every command that resolves the config needs them, not only
`gym eval run`: `gym eval prepare` and `gym eval plan` refuse the same unset values, so a submitted job passes the
six overrides to its prepare step as well.

| Key | Meaning | On every model call |
| --- | --- | --- |
| `context_window` | served context window in tokens | NOOA's context planner budget |
| `max_output_tokens` | reply cap | `max_output_tokens`, unless the benchmark request caps the reply itself |
| `temperature`, `top_p` | sampling | sent unless the benchmark request sets them |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high` or `max` | `reasoning.effort`, unless the request sets `reasoning`; `none` sends nothing |
| `media_inputs` | media the model accepts: `image`, `audio`, `video`, `pdf`; `[]` is text only | media the model does not accept is replaced with NOOA's short note before the request is built |

A benchmark's own `responses_create_params` (temperature, top_p, max_output_tokens, reasoning) win over the declared
values, so Gym's run settings keep their meaning. For GLM 5.3 served by vLLM that is, for example,
`context_window: 1048576`, `max_output_tokens: 131072`, `temperature: 1.0`, `top_p: 0.95`, `reasoning_effort: none`
(thinking is set on the model server) and `media_inputs: []`.

## Sandboxed execution

`nooa.execution_mode: sandboxed` runs the complete NOOA agent in the task sandbox supplied by Resources.
NOOA stages its own runtime and closes its worker before disconnecting; Resources owns sandbox creation and teardown.
The runtime packages the launching Gym checkout with its unchanged root `pyproject.toml` and full core dependency set.
Tracked package source and assets are copied with their current contents; untracked files, Git metadata, and caches are excluded.
The controller therefore needs a Git checkout and Git installed. The package is installed into NOOA's private Python,
leaving the task's Python environment unchanged. Offline wheelhouses must be rebuilt whenever these packaged bytes change.
Agent and resource sessions use Gym's existing `single_agent_turn` environment. Agent close returns the evidence and final
Resources cookies before verification; Resources closes its sandbox afterward.

`sandbox_runner_timeout_seconds` sets a finite worker ceiling (default: 21420 seconds). Request metadata
`agent_timeout_sec` can shorten that ceiling. This is separate from the environment's episode timeout;
a longer episode timeout does not extend the worker ceiling. The default worker ceiling is below the
default 21600-second episode timeout. The episode budget covers seeding, runtime installation, agent execution,
agent close, and verification, while the worker ceiling starts after seeding. The episode deadline can therefore
fire first. Final cleanup runs after the episode deadline under its own `cleanup_timeout_seconds` budget per callback.
`session_close_timeout_seconds` budgets each cleanup/capture phase and the remaining activation join
(default: 30 seconds per phase). Both settings must be positive and finite.

See [NOOA sessions and SWE-bench Pro](../../fern/versions/latest/pages/evaluation-tutorials/nooa-swe-bench-pro.mdx)
for setup, recipes, service lifecycle and observation limits.

Sandboxed mode requires the Resources server to provide `sandbox_access`; otherwise seeding returns HTTP 422.
Its provider must support `AsyncSandbox.connect`. The Gym model server address must be reachable from inside
the task sandbox: configure `use_absolute_ip` / `default_host` for the deployment instead of a controller-only
loopback address. Harbor runs (`gym eval run <target> --agent <name>`) set `use_absolute_ip` for the model server
automatically, so this applies only outside Harbor. Runtime packaging runs in a thread so seeding does not block
other agent requests.

Close raises if the worker stop is unconfirmed or the remaining activation cannot be joined within its budget;
the session stays available for a close retry. The bounded join also applies to embedded mode. An activation
interrupted by session close returns HTTP 409 to a caller whose request was not itself cancelled.

### NOOA trace files

Every rollout keeps a NOOA trace: the worker routes NOOA's OpenTelemetry spans and message journal to one JSONL
file per NOOA session (an Atom root and its children are separate sessions) and joins them into `traces.jsonl`
beside `result.json` before the result is published. The runner copies that file to
`<artifacts_dir>/<rollout_id>/traces.jsonl` on the agent server host (`artifacts_dir` defaults to
`nooa-artifacts/`, gitignored). The journal records, for each model call, the exact messages sent and the reply
received as content-addressed blocks, plus the Gym response id, so any call in the rollout record can be
reconstructed from the trace of the rollout named by its `rollout_id`; `nooa import-traces` loads the file into
NOOA's viewer as is. Gym's rollout record remains the evidence that grading, export and training use. A worker
that dies before flushing leaves no file: the rollout is collected as usual and its observations carry a
`trace_unavailable` gap. Embedded runs enable the same exporter once in the agent server process and write under
`<artifacts_dir>/embedded/`.


## Sandbox worker supervision

`sandbox_runner_timeout_seconds` bounds the worker execution budget; task metadata can shorten it. Exhausting that
budget ends the rollout where it stands: the last checkpoint is returned as an `incomplete` response with termination
reason `timeout_exceeded`, and the benchmark grades it (Terminal-Bench scores it as a miss). It is never retried as a
model or provider outage. A worker that leaves no checkpoint at all raises a terminal `agent_timeout` failure instead,
which Gym records without retrying.
Partial checkpoints remain available for failure inspection; they never authorize grading. Transient artifact
transfer failures receive one retry within the existing capture deadline. Worker startup and supervisor failures
retain up to 2,000 characters of output-log diagnostics after artifact capture.
