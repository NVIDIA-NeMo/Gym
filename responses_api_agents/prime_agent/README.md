# Prime Agent

Runs [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) in headless JSON mode.

On the host path, Prime Agent must be on `PATH`. The adapter installs `prime_agent_version` with
the official installer on the first host request when the default command is missing; agent
sessions install Prime Agent inside the task sandbox instead (see below).

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
gym env start --environment prime_agent_math --model-type openai_model

gym eval run --no-serve --agent prime_agent_math_agent \
  --input environments/prime_agent_math/data/example.jsonl \
  --output environments/prime_agent_math/data/example_rollouts.jsonl \
  --limit 5
```

Math and reasoning gym example environments are in `environments/prime_agent_math` and
`environments/prime_agent_reasoning_gym`.

On the host path, each request uses an isolated `HOME` and one-shot Prime Agent worker. `kernel_venv` is shared so
IPython dependencies are reused.

`model` uses `<provider>/<model-id>`. Define the provider in `models_config` or set `model_server`
to use a Gym model server. See `configs/prime_agent.yaml` for all options.

## Agent sessions in task sandboxes

Capability level: Evaluation. A real-model smoke on SWE-bench Pro scored 3 of 3 tasks and resolved 2.

[configs/prime_agent_sandboxed.yaml](configs/prime_agent_sandboxed.yaml) runs Prime Agent **0.7.0**
inside the task sandbox of a `single_agent_turn` Environment Server, for benchmarks whose Resources
Server supplies `SandboxAccess`, such as SWE-bench Pro. Training token-ID capture is off by default. With
`model_server`, model calls go to the Gym Model Server under the rollout's prefix, so capture needs only run-level
`token_id_capture.enabled: true` and either the agent's `token_id_capture: true` or run-level
`token_id_capture.all_agents: true`.

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/prime_agent/configs/prime_agent_sandboxed.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: prime_agent
      resources_tool_transports: []
```

- **Runtime.** Session setup installs private Node.js 22.23.2 and Prime Agent 0.7.0 under
  `/tmp/nemo-gym-prime-runtime-0.7.0-node-22.23.2`. It reuses a runtime only when its recipe fingerprint,
  reported version and kernel import checks match. The fingerprint covers the dependency locks, installer,
  staging helper and pinned runtime identities, so an older unlocked cache is rebuilt.
  The agent server checks the official Node, Python, uv 0.10.12 and four Prime release archive SHA-256 values.
  It localizes the Prime release dependencies to those verified uploads. npm uses
  `ci --install-strategy=shallow --ignore-scripts`, preserving the SDK's nested import paths; every registry
  artifact has an exact version and SHA512 integrity in `runtime/package-lock.json`. The local archive
  integrities are bound to the verified, rewritten uploads. The sandbox checks every uploaded input before
  installation and preserves Node's bundled npm and corepack. Prime runs with its private Node, keeping the
  task's Node and `PATH` intact. Setup needs a glibc image, bash, `sha256sum`, `tar`, `gzip`, `flock`, `mktemp`,
  `cmp`, and access to the npm registry for the locked artifacts.
- **Kernel.** Prime Agent's only built-in tool is an IPython kernel. Setup installs a private Python under
  the runtime's `kernel` directory and the verified release's `prime-agent-runtime`, then sets
  `PRIME_AGENT_KERNEL_PYTHON`. `runtime/python-constraints.txt` fixes runtime dependency versions and
  `runtime/python-build-constraints.txt` fixes the isolated build dependencies. uv applies both files;
  Python package artifacts still come from the index and are not individually hash-locked. The sandbox
  needs access to PyPI. The kernel takes about 440 MB. IPython and Jupyter may write under the task's `HOME`.
- **Network.** A sandbox the agent creates keeps the provider's egress: `network_access` accepts only
  `inherit`, since setup needs the npm registry and PyPI. The Prime Agent release archives come from
  GitHub through the agent server, not the sandbox. A borrowed sandbox keeps its lender's network.
- **Isolation.** Models, tools and state live under `PRIME_AGENT_CODING_AGENT_DIR` in a private
  session directory, so tool shells keep the task's `HOME`. `SandboxAccess.env` is forwarded except
  `PRIME_AGENT_*`, `PI_*` and `OPENAI_*`. Close removes only the session directory.
- **Requests.** The user message is the task and the system or developer message, `instructions`
  and `system_prompt` are appended to the system prompt; both are argv values under 128 KiB. Other
  request fields are refused with HTTP 422. `command`, `extra_args` and `env` overrides, required
  tool grants and a `SandboxAccess` user override are refused at seed.
- **Results.** Output and usage come from Prime Agent's `message_end` events. A model error that
  Prime Agent retries is skipped, and a run whose final message is an error stop fails with the model request's status when Prime Agent reports one (for example
  `model_request_error` 503), a `length` stop is `incomplete`, a supervisor timeout with output is
  `incomplete`, as is a rollout stop at any point. A supervisor timeout before any model reply, a
  nonzero exit or a run without an answer fails. A failed stdout read raises, so the run is retried. Model-call ownership, tool
  timing and child invocations are reported as observation gaps.

## SDK child sessions

`child_app.py` is an evaluation-only entrypoint of this adapter for Prime Agent **0.7.0**
(public revision `be9e2fa0714e7cd1c6bd9bdb1b554d2cc6550387`). It creates a depth-one
child with Prime's own IPython tool and agent-message skill. The parent is a recording
controller; it makes no model calls. This mode requires a borrowed task sandbox and
uses the agent-session API through `single_agent_turn` or `single_agent_turn_legacy`.
The Resources Server prepares the repository and destroys the sandbox after verification.

The implementation calls the public SDK; it does not vendor Prime's source. See
[the pinned SDK](https://github.com/PrimeIntellect-ai/prime-agent/blob/be9e2fa0714e7cd1c6bd9bdb1b554d2cc6550387/packages/coding-agent/src/core/sdk.ts).
The child reuses `PrimeAgent`'s seeded-session validation, activation joining, stop and
close lifecycle, with `PrimeSession.install` and `PrimeSession.stage`. Resources owns
repository preparation and lends the sandbox through `LentSandboxes`; the agent
releases its connection, while Resources stops the sandbox after verification.
The SDK-specific code is limited to its recording parent, execution hooks and capture.

The shared installer supplies Node, Prime and the Python kernel. The SDK entrypoint is
`RUNTIME/lib/node_modules/prime-agent/dist/index.js`, and its AI dependency is loaded
from that package's nested `node_modules`. Before starting the child, a no-model
capability probe imports the installed SDK and AI, checks their versions/public APIs,
discovers the bundled `agent-message` Python skill, and imports its actual Python
helper with the configured kernel. The verified skill's `src` directory is on
`PYTHONPATH`; there is no separate unpinned skill build. Failed probes fail the
activation before a policy request. Probe evidence is retained as `runtime-check.json`.

Child configuration uses `model_server`, `model`, `prime_agent_version`,
`context_window`, `max_output_tokens` (per model call), `max_turns` (complete turns,
including the last turn's tools), `temperature`, `timeout`,
`sandbox_install_timeout_seconds`, and `session_close_timeout_seconds`.
`capture_directory` exports controller receipts into the Resources-owned sandbox;
`artifact_directory` retains captures on the Agent Server. Task Python packages
belong to Resources' `task_python_packages` and `task_python_package_hashes`: exact
versions and SHA-256 wheel hashes, installed with no dependency resolution into a
task-only pure-Python directory. Its path is passed through ordinary SandboxAccess
`PYTHONPATH`. The old child `kernel_packages` setting is rejected.

The child accepts the stock PrimeAgent text-input contract and timeout controls;
request-level sampling, tools, supplied multi-turn history, sandbox user overrides,
continuation and required external tool grants are rejected. Task environment values
use the shared staging filters; only Gym's generated `nemo` provider is uploaded.
The pinned SDK uses thinking off, and supervised rollout stops follow the shared
Prime lifecycle. Session files stay outside the task repository and are removed at
close; the runtime cache lives until the Resources-owned sandbox is destroyed.

The shared installer integrity-locks npm artifacts, constrains Python runtime and isolated
build dependency versions, and checksum-checks Node, Python, uv and the primary Prime release
archives. Task wheels also retain their exact versions and hashes. These controls address
observed npm version drift; they do not establish bit-for-bit reproducibility of an entire
task environment. Python wheel/build artifacts are version-constrained without individual
hashes, and task setup package-manager state and an unpinned task-image tag can still vary.
Keep this entrypoint experimental and record the resolved image and installed inventories.

All model requests use Gym's rollout-prefixed model URL. Until the parent receives
a report, the runtime forces the registered IPython tool; after delivery it allows
an automatic final response. Compaction, auto-refinement, and retries are disabled.
The 65,536-token output reservation shrinks to fit remaining context with an
8,192-token margin. Do not remove or override that per-call cap in the Model Server.

Capability level: **Evaluation with native observation export**. Close exports captured SDK turns, model
response IDs, tool outcomes and supervised process metadata; unavailable evidence remains an explicit gap.
This mode claims neither training nor full trajectory conformance. Raw SDK events, controller
receipts, effective prompts/tools, model requests, and supervisor cleanup receipts
are saved. Function-calling shell probes do not apply: the only exposed tool is
IPython. Terminal-Bench support is not claimed. The first pairing is Partner IF's
repository/report task. A second pairing and runtime reproducibility checks remain
required before this experimental entrypoint is ready to merge. The stock CLI's
SWE-bench smoke described above exercises a different execution path and does not
satisfy the SDK child's second-pairing requirement. The capability probe checks the
installed SDK and `agent-message` interfaces, while regression tests cover provider
credential filtering and the shared session lifecycle. These checks do not establish
a real-model second pairing or eliminate the remaining artifact and image limits described
above; retain this warning until those results are recorded for the tested revision.

Session close exports the native SDK conversation as Gym observations. Kernel errors remain failed tool
observations even when the SDK's outer isError flag is false; missing model response IDs remain evidence gaps.
The borrowed sandbox's provider/ID and cleanup receipt describe supervised process completion. Agent close
does not destroy that sandbox; teardown remains the Resources Server's responsibility.
Well-formed SDK captures are retained even when execution or disposal fails.
Those invocations remain failed, with missing diagnostic files recorded as gaps.
Corroborated parent deliveries can still be verified; absent or malformed captures
remain unscorable.
