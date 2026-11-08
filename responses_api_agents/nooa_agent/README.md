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

The project pins `nooa`, `nooa-cli`, `nooa-bench`, and `nooa-atom` to the same source revision. The CLI supplies
executor support, and the benchmark package supplies the default BenchAgent. Gym's server installation uses
the same project and lockfile.

## Atom

Load `configs/nooa_agent.yaml` and `configs/nooa_atom_agent.yaml`, then select `nooa_atom_agent` in the taskset's
agent route. This constructs `nooa_atom.gym:AtomHarness` and invokes it through
`responses_api_agents.nooa_agent.atom_adapter:invoke_atom`. Each rollout gets one Atom session and Gym's model
client. `nooa.init_kwargs.turn_method` defaults to `handle_batch`; `handle_orchestrate` is also supported by
the pinned harness. Select `nooa.execution_mode: sandboxed` when the task needs its Resources-owned workspace.

Atom's structured return is kept as a verifier-facing message. That synthetic message has no training tokens.
The response metadata separately declares the last served response from the first model-calling root agent,
including its later session turns. Pure orchestration wrappers and later calls from child agents do not
replace that declaration. Environment results carry it to RL's receipt consumer; a declaration that cannot
be matched to a captured response remains masked.

## Embedded CodeAct scope

Embedded CodeAct persists for the duration of one invocation. `execute_python` is not computation-only: it can
read and modify files, start processes, use the network, and change other node state. Embedded mode provides no
sandbox protection. Only injected resource methods are Gym/verifier-observable and authoritative; direct CodeAct
side effects belong to NOOA and are not reliably visible to verification or replay. The trajectory distinguishes
the outer `execute_python` call from nested resource calls.

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
loopback address. Runtime packaging runs in a thread so seeding does not block other agent requests.

Close raises if the worker stop is unconfirmed or the remaining activation cannot be joined within its budget;
the session stays available for a close retry. The bounded join also applies to embedded mode. An activation
interrupted by session close returns HTTP 409 to a caller whose request was not itself cancelled.


## Sandbox worker supervision

`sandbox_runner_timeout_seconds` bounds the worker execution budget; task metadata can shorten it. Exhausting that
budget is an ungradable, non-retryable execution failure, rather than a retryable model or provider outage.
Partial checkpoints remain available for failure inspection; they never authorize grading. Transient artifact
transfer failures receive one retry within the existing capture deadline. Worker startup and supervisor failures
retain up to 2,000 characters of output-log diagnostics after artifact capture.
