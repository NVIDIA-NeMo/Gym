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
  # optional: ++agent.responses_api_agents.nooa_agent.nooa.init_kwargs.render_config.live_context.kind=dynamic
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
  # optional: ++agent.responses_api_agents.nooa_agent.nooa.init_kwargs.render_config.live_context.kind=dynamic
```

Atom's session wraps a strategy failure in `TurnFailedError`; the adapter re-raises the wrapped error so the runner's
termination policy sees the same CodeAct stops and model rejections it sees from any other NOOA agent. Gym resource
tools are bound on the harness class, not on Atom's session agent.

The response ID names the last served model response from the root agent. A later child or summary call does not
replace it. The adapter keeps Atom's structured return as a synthetic assistant message for text verifiers.
After verification, the generic `SingleAgentTurnResult` contract retains the graded response in `verifier_response`
and leaves only model-authored actions in `response.output`. Synthetic messages carry `synthetic: true`; ordinary
output keeps its existing wire shape. Strict capture matching still checks the exact model-authored items and IDs.

NOOA at or above the pinned revision records live context (`<context>`: to-do list, locals, workspace state) as
`LiveContextSnapshot` events in the message history when `RenderConfig.live_context` is `append` (NOOA's default:
a snapshot every 5 turns, only when the content changed). Each model request is then a prefix-extension of the one
before it, so token capture links consecutive calls without any bridge-side replay of earlier requests. The policy is
set per run on the NOOA agent, for `AtomHarness` and `BenchAgent` alike:

```bash
++agent.responses_api_agents.nooa_agent.nooa.init_kwargs.render_config.live_context.kind=dynamic|off|append
++agent.responses_api_agents.nooa_agent.nooa.init_kwargs.render_config.live_context.every_n_turns=5
```

`dynamic` regenerates the context block on every call (NOOA's behaviour before this pin; each request rewrites the
prompt suffix), `off` sends none, and `append` records it in history as above. Evaluations that are compared with
earlier `dynamic` runs set `dynamic` explicitly. Benchmark `instructions` and leading system/developer text precede
NOOA's own prompt on every model request, including wrapper and child calls. Atom's `solve()` receives the user task
separately. Non-text system/developer instructions are rejected explicitly.

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

`nooa.execution_mode: sandboxed` borrows the task sandbox supplied by Resources when present. Otherwise it creates a
sandbox using `sandbox_provider` and `sandbox_config` (including the task image); absent either a borrowed sandbox or
a configured provider, seeding fails before inference. NOOA closes its worker and disconnects from borrowed sandboxes;
Resources owns their teardown. NOOA stops its own sandboxes, including after failed or cancelled runtime setup.

`network_access` controls agent-owned sandboxes: `inherit` keeps provider defaults, `model_only` allows the model
host, and `model_and_tools` also allows direct HTTP tool hosts. Restricted modes require an offline runtime bundle
and a provider that enforces network policies. Required tool grants blocked by `model_only`, conflicting explicit
network policies, unsupported providers and restricted settings on borrowed sandboxes are rejected before creation.
MCP grants remain a separate capability; this change does not add MCP transport support.

When `runtime_wheelhouse` is supplied, its wheels, Python archive, source identity and platform define the worker
runtime. The controller uses the bundle directly and needs neither Git nor a checkout. It validates the bundle's
existing platform, glibc and artifact digests and keeps the source-plus-wheel identity in the runtime path. Supplying
a bundle does not update its worker code from the controller: prepare a new bundle to adopt worker source changes.

Without a supplied bundle, the runtime packages the launching Gym checkout with its unchanged root `pyproject.toml`
and full core dependency set. Tracked package source and assets are copied with their current contents; untracked
files, Git metadata and caches are excluded. This source-building path still requires Git and a checkout. Both paths
install into NOOA's private Python, leaving the task's Python environment unchanged.
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

Borrowed mode requires a provider supporting `AsyncSandbox.connect`. The Gym model server address must be reachable from inside
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
