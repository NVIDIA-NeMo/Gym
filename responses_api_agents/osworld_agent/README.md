# OSWorld Responses API Agent

The OSWorld agent runs complete desktop-computer tasks through NeMo Gym. The
agent and benchmark stay in the Gym process; only the OSWorld desktop
container/VM lifecycle moves into Gym Docker Sandbox. Each `/run` request
creates an OSWorld `DesktopEnv`, sends observations to the configured model,
parses and executes actions, invokes OSWorld's inline evaluator, and returns
the trajectory and reward in Gym's Responses API shape.

This directory owns the reusable runtime. Dataset preparation, benchmark
configuration, model-specific overlays, serving recipes, and the full user
guide live in the [OSWorld benchmark](../../benchmarks/osworld/README.md).

The OSWorld agent and its vLLM transport use a managed Python interpreter at
the repository's declared Python floor. This matters when Gym is loaded from
a newer checkout inside an older accepted container: the parent CLI may keep
running there, while each isolated server venv is resolved with the compatible
managed interpreter. For offline runs, pre-seed uv's Python install directory
and export `UV_PYTHON_INSTALL_DIR` before `gym env prefetch`.
If the parent Ray cluster is still on a different 3.13 patch, the operator must
either upgrade the whole cluster or explicitly set
`RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor`; Ray otherwise requires an
exact patch match. The relaxed mode is only valid within one Python minor and
must be covered by a real server-registration smoke, not assumed from a
successful resolver.

## Request and response contract

The rollout collector sends the complete upstream task under
`verifier_metadata.osworld_task`. The runtime passes that task to
`DesktopEnv.reset(task_config=...)` without translating its setup or evaluator
semantics. `responses_create_params` supplies per-rollout sampling overrides.

A completed response includes:

- Gym `reward`, using `binary` (default), `raw`, or the opt-in `signed_v1`
  mapping described below;
- `runtime_eligible` plus `runtime_admission_reason` and policy identity;
  `mask_sample` remains the backward-compatible inverse carrier for
  infrastructure, timeout, and evaluator failures;
- `horizon_reached` and `evaluation_completed`, so an evaluated `max_steps`
  outcome is not confused with corrupt runtime data;
- `verifier_metadata.osworld_score`, `osworld_steps`, completion/error state,
  termination reason, model identity, artifact directory, and proxy provenance;
- a schema-v2 `trajectory_contract` and one semantic `(state, action, reward,
  next_state, done)` transition per environment step;
- `trajectory_model_calls`, preserving each materialized prompt, sampled
  action, reward/done linkage, parser outcome, and any available token/logprob
  evidence. Screenshot bytes live once in `media_assets`; prompts reference
  them by ordered `media_id`.

### Semantic trajectory and exact model-call evidence

Trajectory collection is automatic; it is not a training mode. Every runner,
including closed model APIs that do not expose tokens, returns the semantic
contract. `trajectory_contract.capabilities` says which stronger evidence is
available.

For endpoints that return `prompt_token_ids`, `generation_token_ids`, and
`generation_log_probs`, Gym additionally emits `context_compaction_contract`
exact authority. Each materialized model call is independent, so successive
prompts may rewrite any earlier token or media position. Parser retries are
also separate model calls and are not collapsed into the environment step.
NeMo-RL can therefore reconstruct prefix-contiguous physical traces while one
logical rollout retains one reward and advantage.

When training-token capture owns the token arrays, the agent may not receive
exact token evidence itself. In that case, `response.output` preserves one
assistant message per **accepted** model call, including the original thinking
text. A successful retry is included; parser-rejected attempts are not accepted
history. Complete `trajectory_model_calls` evidence still preserves every
attempt, its raw completion, acceptance flag, and parse error. Calls without a
response or with empty served content
(such as a local context-overflow refusal) provide no usable returned-content
witness and are skipped; the capture ledger independently records what ran.
An empty accepted history stays empty instead of falling back
to synthesized step text. `terminal_response_id` identifies the last observed
non-empty served response by its Chat completion ID. This lets the
trainer join returned history to capture records without matching against
parser-normalized actions. Capture-disabled responses and the existing exact
per-call token-evidence path keep their previous behavior; that compatibility
path still exports all exact calls, including parser retries.

With NeMo-RL capture selection, `returned_history` selects the accepted output;
`all_captured` explicitly opts into training captured rejected attempts too,
subject to the trainer's completion and evidence checks. Neither capture nor
the presence of a call in diagnostic evidence alone makes it accepted history.
The scored terminal and selected ancestry must still be valid: if the last
served call was rejected, its ID is not reassigned to an earlier accepted call.
`returned_history` rejects that selection because the terminal is absent from
accepted output. If a rejected terminal lacks that ID, response construction
fails rather than permitting fallback to an earlier call. Likewise, if retry
feedback makes an accepted call depend on
a rejected captured parent, the trainer must reject the non-closed selection,
not silently add that parent to the loss. Use an explicit compatible selection
policy for these cases. This agent does not rewrite prompts or fabricate new
capture roots to work around selection validation. Gym's runtime admission,
evaluator score, and reward remain independent of this training selection.

Training manifests should supply a model-independent caller-owned identity:

```json
{
  "trajectory_identity": {
    "schema_version": 1,
    "group_id": "chrome-task-001",
    "task_id": "task-001",
    "rollout_index": 0,
    "attempt_index": 0
  }
}
```

The trace-aware NeMo-RL launcher derives and stamps `rollout_id` inside that
object and binds a runtime generation contract before training dispatch.
Standalone benchmarks derive a stable identity automatically and still emit
the same semantic contract. A trainer must fail closed unless the identity is
caller-owned, exact evidence is complete, runtime admission is valid, and its
own tokenizer/template/processor contract passes. Gym exposes the first two
admission layers independently: `trajectory_contract.runtime_admission`
classifies VM/evaluator trust, while `exact_trace_admission` reports evidence
facts and leaves the final loss/token decision to the training consumer. The
legacy `training_eligibility` and `eligible` fields remain compatibility views.
The `context_compaction_contract` wire name is retained for compatibility with
the existing NeMo-RL physical-trace reconstructor; it is evidence capability,
not a Gym training switch.

At `max_steps`, the runner executes the last valid action first, evaluates the
resulting VM state, preserves score/reward, records
`termination_reason=max_steps` and `horizon_reached=true`, and then applies
runtime admission. Both reward-zero and reward-one horizons remain
runtime-eligible when evaluation completed. The model adapter never fabricates
`FAIL` and never decides `mask_sample`; it reports parse and transport facts.

OSWorld continues to evaluate inside `env.evaluate()`. The environment backend
is selectable between OSWorld's provider directly and Gym Sandbox. In the
Sandbox path, OSWorld still owns `DesktopEnv`, setup, controllers, actions,
and evaluators; Gym Sandbox owns only the VM container lifecycle, dynamic
service endpoints, status, and cleanup.

### Generated-task evaluators and reward modes

Generated tasks can carry a `taskgen_plugins` block with schema
`taskgen-plugins/1`: getter/metric names, Python source under
`modules.getters` and `modules.metrics`, and a SHA-256 digest of those modules.
`taskgen_plugins.register_task_plugins()` validates the block and registers
the declared functions before `DesktopEnv.reset()` resolves them. It refuses
to replace built-in OSWorld functions or reuse a name with a different digest
within one task. Registrations last only for that task: `run_osworld_task()`
clears them after closing the environment in its `finally` block, including
partial registration, reset, rollout, evaluator, and environment-close failures.
Cleanup removes only the exact callable objects installed by the loader,
including aliases/partials, never built-ins or later replacements. Reused Ray
workers therefore do not carry evaluator functions into the next task; different
tasks may use the same name for different implementations or share a getter
with different metrics. Tasks without this block are unchanged. The remote Resources
Server path rejects these plugins because it cannot register their functions
in the evaluator process.

Plugin source executes as Python in the agent's evaluator process. Only use
trusted task datasets; the digest checks content consistency, not authorship
or sandbox isolation. This mechanism does not rewrite canonical OSWorld task
setup or evaluator definitions.

`reward_mode` maps the evaluator score and the agent's terminal action:

| Mode | Reward |
| --- | --- |
| `binary` (default) | 1 when the evaluator score reaches 1.0; otherwise 0 |
| `raw` | The evaluator score, including partial credit |
| `signed_v1` (opt-in) | 1 on evaluator success; -1 for `DONE` without success; otherwise 0 |

An infeasible task whose evaluator awards success for `FAIL` still earns 1.
Infrastructure admission remains separate from reward mapping. `signed_v1`
does not penalize reaching the step limit without `DONE`, so it can encourage
continuing to the horizon; it is not a new benchmark default. Aggregate metrics
continue to use the evaluator's raw score and binary success, independently
of the selected training reward mapping.

## Runtime components

| File | Responsibility |
| --- | --- |
| `app.py` | Gym server, request validation, model transport, Ray dispatch, response and aggregate metrics |
| `client.py` | `DesktopEnv` lifecycle, cache staging, action execution, evaluation, logging, and artifacts |
| `rollout_outcome.py` | Termination classification and runtime admission; never changes actions, score, or reward |
| `runner_registry.py` | Runner names, upstream class paths, and default observation/action contracts |
| `adapter_agents.py` | Gym-owned model scaffolds, including `NemotronV3NanoOmniAgent` |
| `model_protocol.py` | Model wire contracts and the selected-history message renderer interface |
| `super35_protocol.py` | Super 3.5 VL tool-XML parsing, message rendering, and request options |
| `trajectory.py` | Model-independent semantic trajectory identity, transitions, and evidence capabilities |
| `exact_trace.py` | Optional exact model-call/token/media evidence for trace-aware trainers |
| `action_parser.py` | Gym pyautogui/control-action parsing and validation |
| `proxy.py` | Explicit proxy-task configuration validation and non-secret provenance |
| `runtime_dependencies.py` | Version/import readiness check and explicit-install remediation for excluded packages |
| `taskgen_plugins.py` | Validate, register, and clear task-scoped trusted evaluator functions without replacing OSWorld built-ins |
| `reward_modes.py` | Map evaluator score and terminal action to the configured training reward |
| `sandbox_desktop_env.py` | Scoped `DesktopEnv` compatibility wiring for the Gym Sandbox backend |
| `sandbox_provider.py` | OSWorld provider contract backed by Gym Sandbox lifecycle and endpoints |

### Shared Gym foundations

The following foundations are already merged into Gym `main`:

| Change | Contract used here |
| --- | --- |
| [#3591](https://github.com/NVIDIA-NeMo/Gym/pull/3591) | Isolated component runtimes, setup configuration, and rollout identity |
| [#4128](https://github.com/NVIDIA-NeMo/Gym/pull/4128) | Shared training-token capture, lineage, and external-staging contracts |
| [#4822](https://github.com/NVIDIA-NeMo/Gym/pull/4822) | Docker-create reconciliation and E2B service endpoints/worker transport |

The taskgen, reward-mode, and served-history changes in
[#4896](https://github.com/NVIDIA-NeMo/Gym/pull/4896) use these merged
foundations; no additional stacked PR is needed for that scope.
[#3368](https://github.com/NVIDIA-NeMo/Gym/pull/3368) is the separate optional
simple-agent compaction migration, not OSWorld's `history_policy` implementation.
[#4823](https://github.com/NVIDIA-NeMo/Gym/pull/4823) separately fixes shared
HTTP-client shutdown and wildcard-host dialing. Neither change is bundled
into #4896 or a prerequisite for merging its OSWorld-only changes. Workloads
that need those separate capabilities should track their PRs explicitly.

### OSWorld source dependency

This agent intentionally installs the immutable
[`JeffPengCoder/OSWorld`](https://github.com/JeffPengCoder/OSWorld) fork at
commit `f32ab2b74e3ea66e6a8eb0d87876a12ce93904d5`, as declared in
[`pyproject.toml`](pyproject.toml). That revision starts from upstream
OSWorld `83e85344` and includes the `nv-gym` provider overlay, proxy-runtime
repair, logging hardening, VLC gateway-auth fallback, the per-environment
provider contract, opt-in setup/evaluator return-code semantics, and the
restricted-guest Chrome ownership fix without rewriting canonical OSWorld task
configs. Gym supplies orchestration and the
worker control plane; OSWorld remains independent of Gym.

The fork declares NumPy/OpenCV requirements by Python version: Python 3.12
retains NumPy 1.26/OpenCV 4.8, while Python 3.13 uses NumPy 2.1+ and
NumPy-2-compatible OpenCV 4.10.0.84+. Gym's role-local ranges further select
the supported runtime. No NumPy override is needed to bypass OSWorld metadata;
both this agent and the resources server consume the same source revision.

The dependency is consumed as a commit-addressed source archive so uv does not
initialize optional OSWorld submodules. Gym does not mutate the installed
checkout at runtime, and the adapter does not monkeypatch OSWorld setup
semantics. Update the fork URL or commit only together with contract tests and
a real OSWorld rollout. Do not rewrite task setup to
compensate for adapter behavior; task-corpus changes require their own dataset
authority and evaluation review.

## Supported runners

`runner_name` selects the model-facing scaffold:

| Runner | Ownership and contract |
| --- | --- |
| `gym_pyautogui` | Gym prompt and Python/pyautogui actions |
| `prompt_agent` and `prompt_agent_*` | Upstream OSWorld `PromptAgent` observation/action variants |
| `pointer_agent` | Upstream PointerAgent planner/executor/verifier loop |
| `m3_agent` | Upstream MiniMax M3 scaffold and protocol |
| `nemotron_v3_nano_omni_agent` | Gym-owned Nemotron 3 Nano Omni scaffold and parser |
| `nemotron_super35_agent` | Shared protocol agent with the Super 3.5 VL tool-XML protocol |
| `qwen3_omni_agent` | Upstream Qwen3VL scaffold through Gym model transport |

The benchmark directory contains the model- and runner-specific YAML overlays.
Those examples do not change the generic runtime defaults in this directory.

### Nemotron response contract

The adapter-owned Nemotron parser requires an explicit `## Code` section, so
it never executes an unrelated code block from prose. `## Thought`,
`## Action`, and `## Code` values may begin on the heading line or the next
line, and Code may be fenced or unfenced. Thought and Action are descriptive
metadata; an explicit, syntactically valid Code section remains executable even
when an Action description is absent. Python is syntax-checked before OSWorld
executes it, and terminal actions require an explicit `success` or `failure`
status.

The current model response must contain Code, but maintained conversation
history deliberately retains only Thought and Action. Thinking-mode assistant
history preserves the model's `<think>...</think>` wrapper. Omitting previously
executed Code matches the validated Nano Omni prompt contract and avoids sending
the same executable payload twice; this behavior is part of the standard
`NemotronV3NanoOmniAgent`, not a run-local subclass or import overlay.

The response this parser reads also depends on how the model is served. The
chat template and `nano_v3` reasoning parser for Omni v3 are listed under
[Serve the checkpoint](../../benchmarks/osworld/README.md#serve-the-checkpoint);
other Nemotron generations ship their own tokenizer, template and parser.

When adding or upgrading a model, capture representative lossless responses
and add focused parser regressions for heading placement, fenced and unfenced
Code, literal newline escaping, reasoning/content separation, tool calls, and
terminal status syntax. Supported formats should remain explicit rather than
recovering executable code from arbitrary prose.

### Super 3.5 VL tool XML

Select `benchmarks/osworld/configs/osworld_agent_super35.yaml`, or prepare the
benchmark with `--profile super35`. This uses the Gym-owned
`nemotron-super35-tool-xml-v3` protocol. It requests raw
`<tool_call><function=...><parameter=...>` response content. Supported tools are `exec_py(code_block)`, argument-free
`computer.wait`, and `computer.terminate(status, answer?)`. Mouse positions
must be literal normalized coordinates in `[0, 1]`; Gym projects them to the
screen dimensions before execution.

The profile keeps the initial screenshot plus the latest three screenshots,
up to four images per request, and retains complete assistant tool calls and
thinking in chronological history. Observations are user turns. It passes
`enable_thinking` and `truncate_history_thinking: false` through the policy
transport. The profile enables `forward_request_chat_template_kwargs` on
Gym's vLLM model adapter so those per-request options reach the backend.
The protocol sends JSON-schema `tools` and `tool_choice: none` on every
request, including retries. The model's chat template renders these definitions;
the system message does not duplicate a handcrafted `<tools>` block. For this
model's pinned template, `none` leaves the rendered prompt unchanged and requests
text output so Gym can apply its strict XML parser. `skip_special_tokens: false`
preserves the XML delimiter tokens. The
endpoint must accept the exact dotted tool names `computer.wait` and
`computer.terminate`. A gateway that rejects those names is incompatible with
this profile; its HTTP error is an infrastructure failure, not a model score.

Use a compatible Super 3.5 VL model service with reasoning separation and
automatic tool extraction disabled. The default profile requires at least four
images. The native vLLM 0.32.0 `nemotron_v3` reasoning engine can omit a repeated
`</think>` delimiter from action content. The optional
[`super35_reasoning` serving plugin](../../benchmarks/osworld/tools/README.md#super-35-reasoning-boundary)
preserves the exact suffix after the first boundary, with CPU regression checks
and a version-pinned launch command. It runs only in the serving environment;
Gym's strict action parser remains unchanged. Reasoning-only trailing whitespace
trimming remains explicit and requires retaining original generations for audit.

Omit `--enable-auto-tool-choice` and `--tool-call-parser`. In vLLM 0.32.0,
`qwen3_coder` can extract calls whose original XML is missing a delimiter.
It can also discard tool XML with `tool_choice: none`, so that request option
alone is insufficient while the serving tool parser remains enabled. The agent
rejects returned native `tool_calls` on this raw-content contract. Gate a new
serving version against its original completion or token IDs before scoring.

`computer.wait` uses the runner's
configured post-action delay. Gym still owns the horizon; malformed replies
are bounded parser retries, not synthetic model-authored `FAIL` actions.
Unsupported coordinate modes fail during configuration resolution and direct
agent construction. Only `relative` is supported by this protocol.

The strict local XML/JSON parser and coordinate conversion are unchanged. The v3
transport preserves raw response content and reasoning without fenced-code
recovery or delimiter repair. The parser function retains its existing native-call
support for other explicit callers, but this profile requires raw XML so the
server cannot silently expand its acceptance grammar. Freeze and record the
serving reasoning parser too; retain the pre-parser completion or token IDs
when claiming raw-generation integrity. An API envelope alone cannot prove
what a serving parser consumed. Earlier v1/v2 scores describe different wire
contracts and must retain their original configuration and provenance.

After coordinate projection, the executor passes Python actions as the exact
string argument to `exec(...)`. This host-owned wrapper lets compound statements
run after DesktopEnv's semicolon-separated imports, including through a remote
Resources Server. It does not insert characters into the parsed Python source
or repair syntax. Trajectories retain the parsed action and the actual execution
wrapper separately. Nano Omni and Qwen runner execution remain unchanged.

This is an independently implemented Gym prompt and protocol. It does not
load an external agent repository, change the Nano Omni defaults, or replace
the scaffold pinned by a published recipe. Treat a change of protocol or
history policy as a distinct benchmark configuration; existing scores do
not qualify a different profile automatically.

The R6 experiment used fixed history with three images including the current
screenshot, a 100,000-token output budget including thinking, no MTP, and 128
workers. Those settings are documented with the serving plugin; the checked-in
profile retains its original four-image history and 4,096-token budget.

### Adding a model protocol

`OSWorldProtocolAgent` owns the model-call/retry loop and uses a registered
`OSWorldModelProtocol` for its system prompt, assistant-history template,
parser, optional message renderer, thinking-dependent template options, API
tools, allowed coordinate modes, and execution wrapper.
`NemotronV3NanoOmniAgent` remains a compatibility entrypoint with its existing
Markdown defaults.

Operator commands for switching models and running them concurrently are in
the [benchmark guide](../../benchmarks/osworld/README.md#model-isolation-switching-and-regression).
This section defines the developer contract. There is no automatic protocol
inference from the model name and no registered v4 implementation at present.

#### Ownership and compatibility boundaries

| Layer | Owner and selection | What must remain stable for an existing recipe |
| --- | --- | --- |
| System prompt, instructions and assistant history | A named `OSWorldModelProtocol` in `MODEL_PROTOCOL_REGISTRY` | Prompt/template content and the protocol ID |
| Observation roles and media ordering | Protocol-specific `message_renderer`, when present | Renderer behavior and its versioned ID |
| Action parsing | The selected protocol's `parser` callable | Accepted grammar, rejection behavior, coordinate contract and parser ID |
| Tools and template request options | Protocol fields applied by the shared agent and model transport | Tool schemas, `tool_choice`, `skip_special_tokens`, thinking/template options and response-preservation mode |
| Action execution convention | Protocol `python_execution`; the shared client applies it | The selected `direct` or `exec` convention, parsed source and separately recorded execution wrapper |
| Screenshot selection | Shared history-policy implementation, selected per agent | Policy name/parameters and actual selected image turns; instance state is reset per task |
| Model calls and retries | Shared `OSWorldProtocolAgent` and transport | Retry budget, feedback, deadlines, failure classification and any context-length recovery |
| Desktop lifecycle and reward | Selected backend, OSWorld action/evaluator path and Gym admission logic | Guest identity, task definitions, evaluator semantics and model/infrastructure distinction |
| Tokenizer, chat template and serving reasoning parser | The separately deployed model service | Pinned revisions/digests, parser registration/selection and raw-generation-to-response integrity |

Nano Omni and Super 3.5 have different protocol objects and parser functions.
`NemotronV3NanoOmniAgent` subclasses the shared loop as a compatibility
entrypoint; its default remains `nano-omni-v3-osworld-v1`. The Super runner
selects `nemotron-super35-tool-xml-v3`. Its specialized flags are enabled through
the selected protocol, not by mutating the Nano protocol. Qwen's existing
`qwen3_omni_agent` runner continues to load upstream
`mm_agents.qwen3vl_agent.Qwen3VLAgent` through Gym's transport adapter.

This isolates model-specific choices, but changes to the shared loop,
transport, history implementation or desktop execution can affect multiple
runners. Treat registered protocol objects as constants: add a new registration
for a changed contract, and do not mutate registry entries while experiments
are running. Protocol IDs are recorded identities, not an automatic check of
the remote checkpoint/template. Unknown explicit IDs raise an error; a
misconfigured endpoint serving another model still requires deployment checks.

Protocol resolution uses the explicit agent `model_protocol_id` first, then
the runner's registered default, then the legacy Nano Omni default. New model
profiles should set the ID explicitly. History selection is configured
separately; choosing a protocol does not select an image policy automatically.

#### Implementation steps

For another screenshot-driven model, including a future v4:

1. Freeze the checkpoint revision, tokenizer, chat template, expected model
   response format and serving parser. Capture real original responses for
   successful actions, terminal actions and malformed outputs. Do not assume
   that a new generation uses the previous generation's format.
2. Implement a protocol module, following the interface in
   [`model_protocol.py`](model_protocol.py). Supply `protocol_id`, `parser_id`,
   `instruction_template`, `step_template`, `text_history_template`, both
   system-prompt variants, both assistant-history variants and the parser
   callable. The parser returns `(action_description, actions, parsed_info)`;
   history placeholders must be provided by `parsed_info`. Define the actual
   model grammar explicitly and retain malformed original responses as evidence.
3. Set optional renderer, API tools, template kwargs, coordinate restrictions,
   response-preservation and execution fields only when required by this
   model's contract. The Super 3.5 module is an example of those extension
   points, not a template whose XML grammar and serving flags apply to every
   model. Provide the independent protocol object in
   `MODEL_PROTOCOL_REGISTRY` in `adapter_agents.py`; keep existing entries.
4. Add a distinct entry to `RUNNER_REGISTRY` in `runner_registry.py`, with
   `kind: model_protocol_agent`, agent class
   `responses_api_agents.osworld_agent.adapter_agents.OSWorldProtocolAgent`,
   the new default `model_protocol_id` and model-appropriate `agent_kwargs`.
   Keep protocol and history configuration at the agent-config level, outside
   `agent_kwargs`; configuration resolution rejects those misplaced keys.
5. Add a benchmark YAML profile that selects the runner and explicit protocol
   ID, model endpoint/transport, sampling, budgets and history policy. Wire its
   agent and environment-server instance names consistently. To expose a new
   CLI profile, add it to **both** `PROFILE_CONFIGS` and `PROFILE_AGENT_NAMES`
   in `benchmarks/osworld/prepare.py`, and test the composed configuration.
   Repeated `--config` flags replace the selected profile's config-path list,
   but do not change the agent name chosen by `--profile`; they are not a
   shortcut for registering a differently named model profile.
6. Add behavior tests and perform the real-model checks below. Keep training
   and evaluation contracts aligned, or declare the intended difference using
   the existing parity mechanism. Publish a new score only after freezing the
   complete run configuration; a passing protocol unit test is not a benchmark.

When a new model needs its own planner, multi-stage executor, observation type
or action space beyond the shared screenshot/pyautogui loop, use a separate
runner/adapter. Preserve its own contract, as with the existing Pointer and
Qwen paths, instead of adding model-name branches throughout the shared loop.

An optional renderer receives `ProtocolMessageContext`: the instruction,
system prompt, observations, serialized assistant history, current
observation, and the history policy's selected image indices. It chooses
message roles and media order without changing image selection or execution.
Assistant-history templates may reference fields returned by the parser.
Provide a versioned renderer ID; renderer identity and request options enter
the wire contract. Bump parser/protocol IDs when changing their semantics.
Agents with their own planning or execution loop continue to use the
existing runner extension points.

#### Validation before changing an existing recipe

| Change | Required evidence |
| --- | --- |
| A new protocol or model-specific renderer | Tests with real response fixtures; ordinary actions, all supported terminal statuses, wait behavior, history roles/media order, coordinate boundaries, unsupported tools and malformed output; a representative real model/desktop smoke |
| Shared model-call, retry, history or execution code | Existing Nano, Super 3.5 and affected upstream-runner tests; cross-model request/history assertions and representative smoke rollouts for affected behavior |
| Serving template or reasoning/tool parser | The real pinned serving parser/tokenizer path; original completion text/token IDs compared with the response handed to Gym; documented pre-existing normalization/recovery behavior and whether each path was exercised |
| Preparation/profile defaults | Composed-config tests for each affected profile, including omitted versus explicit sampling overrides, agent instance selection, backend and history policy |
| Documentation only | Check links, command options and configuration examples against current code; no new model inference is required |
| A full benchmark claim | Frozen source, checkpoint, serving stack, task input, guest image, prompt/protocol/parser/history identities, sampling, budgets, concurrency and timeouts; complete per-task outcomes and model/infrastructure failure accounting |

Run the existing agent and preparation tests from the repository root using
the prepared Gym development environment:

```bash
python -m pytest responses_api_agents/osworld_agent/tests benchmarks/osworld/tests/test_prepare.py
pre-commit run --all-files
```

These tests cover the shared execution loop, existing runner contracts and
profile composition. For a change to other benchmark tools, also run their
targeted tests under `benchmarks/osworld/tests`. The optional Super 3.5 vLLM
plugin has a separate [CPU regression command](../../benchmarks/osworld/tools/README.md#super-35-reasoning-boundary)
that runs inside the pinned serving environment; Gym unit tests do not import
or validate that deployment dependency automatically.

For new prompt/configuration experiments, verify action-parser source parity
and exact raw-response-to-parser-input integrity before scoring. Preserve the
original model response, including malformed JSON, XML, Python and missing
delimiters. Do not insert a missing bracket/tag, rewrite tool arguments or
execute a host-repaired response. A bounded retry requests a new model response
and retains both attempts. Existing compatibility paths, including legacy
Markdown handling or a serving parser's reasoning-whitespace presentation,
must be identified, frozen and audited rather than silently broadened.
Neither a step cap nor exhausted parsing retries supplies a model-authored
terminal failure. Keep model failures distinct from infrastructure failures.

Changing a parser/renderer/protocol requires a new applicable identity and
new experiment provenance. Do not hot-patch active experiments, overwrite
completed runs, or reuse an old score for a new checkpoint/profile. The
Super 3.5 R6 result is evidence for its frozen registered-runner/direct-vLLM
deployment. It is not a new full-suite regression of Nano Omni v3, a validation
of an unimplemented v4 model, or an end-to-end test of every Gym service/backend.

### Model protocol and history policy

The protocol agent keeps two independently selectable identities:

- `model_protocol_id` selects prompts, message templates, and response parser;
- `history_policy` selects which completed turns remain live screenshots and
  which are folded into text.

Fixed three-image evaluation is explicit:

```yaml
model_protocol_id: nano-omni-v3-osworld-v1
history_policy:
  name: fixed
  params: {keep_images: 3}
```

The append-stable 3-10-3 training window is a hysteresis policy:

```yaml
history_policy:
  name: hysteresis
  params: {low_water: 3, high_water: 10}
```

An opt-in sink window keeps the earliest screenshots alongside the recent
window; intervening turns remain text in chronological order:

```yaml
history_policy:
  name: sink_window
  params: {sink: 1, low_water: 4, high_water: 4}
```

Here the four live images include one sink image and three recent images.
Equal watermarks produce a sliding window. Setting `low_water: 3` and
`high_water: 10` instead accumulates up to ten images and compacts back to
three, including the sink. The low watermark must exceed `sink` to leave
room for the current observation. Existing fixed/hysteresis policy identities
and normal-path prompt rendering remain unchanged; selecting a sink is an
intentional recipe change, not a default or a guaranteed score improvement.

`snapshot_image_intervals` records the selected half-open turn intervals.
For non-contiguous plans, consumers must use these intervals or per-turn
decisions, not the legacy scalar `image_window_start` accessor. The telemetry
field `snapshot_window_start` describes the trailing interval.

On a context-length rejection, the adapter can shrink the recent-image window
within its existing retry budget, preserving the sink and current observation.
It stops when no smaller valid image set exists. Each actual shrink is recorded
in `prompt_shrink_events`, including on recovered steps; this recovery can
change outcomes relative to the previous unchanged-request retries. Normal
parse failures do not trigger shrinking, and model deadlines propagate to the
runner without parser retries. The adapter reports specific failure kinds and
the terminal attempt's completion fact; runner/runtime admission still owns
masking, and the evaluator still owns reward.

`agent_contract_parity_mode: strict` is the default. It resolves the training
and evaluation profiles at startup and refuses to start if their model
protocol, history policy, or other Gym-owned adapter options differ. To run an
intentional train/eval comparison, set the mode to `declared` and use
`history_policy_by_rollout_purpose`; each response then records the selected
`agent_contract_id`, `history_policy_id`, and `model_protocol_id`.

```yaml
agent_contract_parity_mode: declared
history_policy_by_rollout_purpose:
  training: {name: hysteresis, params: {low_water: 3, high_water: 10}}
  evaluation: {name: fixed, params: {keep_images: 3}}
```

Legacy `max_trajectory_length` and `agent_kwargs.max_live_images` settings are
accepted and normalized to the same identities. The legacy
`agent_kwargs.max_image_history_length` / `max_live_images` fields cannot be
mixed with an explicit `history_policy`; once the explicit form is present,
the top-level `max_trajectory_length` remains only a compatibility field for
other runners. Runtime exact-trace logic still measures the actual token/media
prefix. A policy's structural append expectation never overrides that measured
evidence.

### PromptAgent variants

The registered upstream PromptAgent variants are:

- `prompt_agent_screenshot_pyautogui`
- `prompt_agent_computer_13`
- `prompt_agent_a11y_tree_pyautogui`
- `prompt_agent_a11y_tree_computer_13`
- `prompt_agent_screenshot_a11y_tree_pyautogui`
- `prompt_agent_screenshot_a11y_tree_computer_13`
- `prompt_agent_som_pyautogui`

Runners that need accessibility data enable it when constructing `DesktopEnv`.
Reasoning wrapped in `<think>` or `<thinking>` is removed before actions are
executed.

## Configuration

The base configuration is
[`configs/osworld_agent.yaml`](configs/osworld_agent.yaml). Important fields
are grouped below.

Gym's `skip_venv_if_present: true` explicitly reuses an existing role venv;
it does not refresh dependencies when source manifests change. After a required
dependency change, update that role environment explicitly or run setup with
`skip_venv_if_present: false`. Setup retains the role's `.python-version` and
uv resolver policy, serializes installers, and records its dependency identity
only after installation succeeds. Ordinary source edits do not require setup.

Environment and execution:

- `provider_name`, `container_image`, `headless`, `screen_width`, and
  `screen_height` configure `DesktopEnv`.
- `sandbox_provider` selects a named Gym Sandbox provider configuration;
  `sandbox_spec` supplies the provider-neutral image/resources/entrypoint, and
  `sandbox_vm_path` selects the read-only OSWorld qcow2 base.
  `sandbox_provider_overrides` applies an OSWorld-only recursive delta to the
  selected provider after named configuration resolution. For example, the
  default OpenSandbox delta bounds VM admission retries without shortening the
  shared provider budget used by other Gym workloads.
- `sandbox_require_kvm`, `sandbox_ready_timeout_s`, and
  `sandbox_ready_poll_s` control the OSWorld Sandbox startup gate.
- `concurrency` limits simultaneous `/run` requests.
- `max_steps`, `sleep_after_execution`, `step_timeout`, and `task_timeout`
  bound rollout work. `task_timeout` is the end-to-end Ray attempt deadline,
  covering sandbox creation, environment setup, agent steps, and evaluation;
  it is also checked cooperatively between child steps and applied to Pointer
  model requests. `task_cancel_grace_s` bounds sandbox cleanup before the
  parent force-cancels a worker that remains stuck.
- `cache_dir` is OSWorld's mutable per-run cache; `setup_cache_dir` points to
  the read-only cache populated by benchmark preparation.

Runner and model behavior:

- `runner_name`, `action_space`, and `observation_type` select a registered
  runner contract.
- `env_class_path` and `agent_class_path` allow explicit compatible classes.
- `agent_kwargs` supplies runner-specific constructor options.
- `max_tokens`, `temperature`, and `top_p` provide server defaults; request
  values can override sampling parameters.

Evaluation and operations:

- `reward_mode` is `binary`, `raw`, or opt-in `signed_v1`; aggregate metrics always report both
  binary success and raw OSWorld reward rates over the measured, unmasked
  subset, following Gym's shared aggregation policy. Normal evaluated failures
  with reward zero stay in that subset. Report `coverage/measured_rollouts`,
  `coverage/masked_rollouts`, and task coverage alongside scores when samples
  are masked; the old `osworld/masked_rollout_count` is replaced by these shared
  counters. An entirely masked run reports coverage without inventing a score.
- `evaluator_disable_gpu` prevents evaluator helpers from reserving rollout
  GPU memory.
- `enable_proxy` and `proxy_config_file` apply only to tasks explicitly marked
  `proxy: true`. `allow_direct_proxy_tasks` preserves the benchmark's direct
  fallback by default on local and Gym Sandbox backends; strict training or
  deployment profiles can set it to `false` to mask those tasks instead.
- `asset_input_jsonl` lets server startup idempotently fill missing prepared
  assets before accepting work.

See the benchmark guide for complete field semantics, logging controls, model
recipes, VM requirements, and troubleshooting.

## Running the benchmark

The current Gym CLI commands are:

```bash
cd benchmarks/osworld
python3 prepare.py \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2

# Explicitly opt in to packages excluded from Gym's shipped environments.
# prepare.py prints these commands with the exact configured venv path.
gym env prefetch
bash ../../responses_api_agents/osworld_agent/install_optional_runtime_deps.sh \
  ../../responses_api_agents/osworld_agent/.venv

# Terminal 1: start configured servers.
gym env start

# Terminal 2: collect against those running servers.
gym eval run --no-serve
```

The installer targets only the managed OSWorld agent venv. It does not modify
the system Python, Gym's root venv, the model server, or the OSWorld VM. The
installer synchronizes `runtime/uv.lock` with `uv sync --locked --inexact`,
which preserves the base agent environment while adding the explicitly opted-in
packages. The runtime project binds both Torch wheels to the CPU index. Commit
both runtime manifest and lock when changing these optional packages.

Choose a model-specific agent composition during preparation. For example:

```bash
python3 prepare.py \
  --profile pointer \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2 \
  --policy-base-url https://ANTHROPIC_COMPATIBLE_HOST/v1 \
  --policy-model-name SERVED_OPUS_4_7_MODEL
```

The Docker backend mounts that base as read-only `/System.qcow2`. Reset means
destroying the Sandbox container and recreating it from the base, matching
OSWorld's Docker-provider behavior. Live RAM/device-state snapshots are not
implemented; callers that require them must select a virtualization provider
with an explicit live-snapshot API.

For data selection, host setup, advanced launchers, model-specific examples,
and expected outputs, use the [benchmark README](../../benchmarks/osworld/README.md).

## Licensing

- Gym adapter code: Apache 2.0.
- OSWorld code and task data retain their upstream licenses. See the benchmark
  README and pinned dependency metadata for details.

## Nano4 producer adapter

The opt-in [Nano4 adapter](nano4/README.md) keeps its producer format, parser, execution feedback and coordinate consumer separate from the existing v3/Super 3.5 adapters. Its direct-library entrypoint depends on the shared model-protocol support in PR #5101.
