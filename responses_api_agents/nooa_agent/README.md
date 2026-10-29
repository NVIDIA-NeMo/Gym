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
uv pip install -r requirements.txt
```

The requirements pin all three NOOA packages: `nooa`, `nooa-cli`, and `nooa-bench`. The CLI supplies executor
support, and the benchmark package supplies the default BenchAgent. Gym's server installation uses this same
requirements file.

## Atom

`configs/nooa_atom_agent.yaml` runs Atom (`nooa_atom.gym:AtomHarness` with
`responses_api_agents.nooa_agent.atom_adapter:invoke_atom`). The harness runs one Atom session per rollout
the way the TUI does, with Gym's model client as the only model: the root turn method is
`init_kwargs.turn_method` (`handle_batch` by default: one agent works the task unattended; `handle_orchestrate`
lets the root delegate to child agents, each a model-calling session recorded under its own invocation), the task is the
first prompt, and the session's `Done` outcome carries the `TaskResult`. Gym's instrumentation hooks are carried
into every session loop, so Atom's `python_cell` executions and tool calls are observed like any agent's. Atom
needs the `nooa-atom` package from the pinned NOOA revision; it runs in the task sandbox under
`execution_mode: sandboxed` exactly like BenchAgent. Known limit: Gym resource tools are bound on the harness
class, not on Atom's session agent, so tasks that rely on Resources-provided function tools should use
BenchAgent until the tools are routed into Atom's tool surface.

## Embedded CodeAct scope

Embedded CodeAct persists for the duration of one invocation. `execute_python` is not computation-only: it can
read and modify files, start processes, use the network, and change other node state. Embedded mode provides no
sandbox protection. Only injected resource methods are Gym/verifier-observable and authoritative; direct CodeAct
side effects belong to NOOA and are not reliably visible to verification or replay. The trajectory distinguishes
the outer `execute_python` call from nested resource calls.

## Sandboxed execution

`nooa.execution_mode: sandboxed` runs the complete NOOA agent in the task sandbox supplied by Resources.
NOOA stages its own runtime and closes its worker before disconnecting; Resources owns sandbox creation and teardown.
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
loopback address. Runtime packaging runs in a thread so seeding does not block other agent requests.

Close raises if the worker stop is unconfirmed or the remaining activation cannot be joined within its budget;
the session stays available for a close retry. The bounded join also applies to embedded mode. An activation
interrupted by session close returns HTTP 409 to a caller whose request was not itself cancelled.

### NOOA trace files

`trace_jsonl: true` makes the worker route NOOA's OpenTelemetry spans to OTLP JSONL files (one per NOOA session,
so an Atom root and its children are separate files) and join them into `traces.jsonl` beside `result.json` before
the result is published. The runner copies that file to `<artifacts_dir>/<rollout_id>/traces.jsonl` on the agent
server host (`artifacts_dir` defaults to `nooa-artifacts/`, gitignored). It is a debugging artifact for reading
what the harness did inside the sandbox; Gym's model-call capture and the rollout's observation bundle remain the
evidence that grading, export and training use. A worker that dies before flushing leaves no file and the rollout
is collected as usual.


## Sandbox worker supervision

`sandbox_runner_timeout_seconds` bounds the worker execution budget; task metadata can shorten it. Exhausting that
budget is an ungradable, non-retryable execution failure, rather than a retryable model or provider outage.
Partial checkpoints remain available for failure inspection; they never authorize grading. Transient artifact
transfer failures receive one retry within the existing capture deadline. Worker startup and supervisor failures
retain up to 2,000 characters of output-log diagnostics after artifact capture.
