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

A Harbor task can declare tools of its own: `[[environment.mcp_servers]]` and `skills_dir`, which the `harbor`
environment server leaves in `/tmp/.nemo-gym/task.json` in the borrowed sandbox. A sandboxed session reads that file
at seed and declares `supports_task_tool_context`, so the environment server gives it such a task. The worker connects
to each MCP server of the task once to list its tools (a `url` server over streamable HTTP, or SSE when the task says
so; a `stdio` server started in the sandbox) and puts them on the agent as one tool object per server, under the
server's name as a Python identifier (`self.playwright.browser_navigate(...)`). A server that does not answer fails the
run as a dependency outage. NOOA's own MCP client opens a session for every tool call; the task's tools keep one
session from the first call to the end of the run, so a server that holds state per session, such as a browser, keeps
it. NOOA has no loader for `SKILL.md` folders, so `skills_dir` is named in a `task_skills` context block of the prompt.
A declaration NOOA cannot be given (a server without a name or endpoint, a name that is not a free agent attribute, a
relative `skills_dir`) refuses the seed, and a file that cannot be read fails it, rather than run the task without its
tools. Embedded mode does not declare the capability.

Close raises if the worker stop is unconfirmed or the remaining activation cannot be joined within its budget;
the session stays available for a close retry. The bounded join also applies to embedded mode. An activation
interrupted by session close returns HTTP 409 to a caller whose request was not itself cancelled.


## Sandbox worker supervision

`sandbox_runner_timeout_seconds` bounds the worker execution budget; task metadata can shorten it. Exhausting that
budget is a time limit, not a retryable model or provider outage. When the supervisor confirms that it stopped the
worker at the deadline, the work the worker saved (the result it publishes when stopped, or else its last checkpoint) is returned as an
`incomplete` response with the `timeout_exceeded` reason, and the environment grades the task as it stands, as it does
for other harnesses' wall-time limits. A deadline that left no saved response, or evidence that could not be read in
full, stays an ungraded, non-retryable execution failure. A checkpoint recovered after any other worker exit remains
available for failure inspection and never authorizes grading. Transient artifact
transfer failures receive one retry within the existing capture deadline. Worker startup and supervisor failures
retain up to 2,000 characters of output-log diagnostics after artifact capture.
