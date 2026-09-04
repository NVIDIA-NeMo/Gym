# Legal Agent Bench Configurable Agent

This task-driven Gym agent runs a configured Gym Responses API agent inside a
Legal Agent Bench (LAB) sandbox. It keeps the benchmark task set, skills, and
verifier fixed while allowing the agent harness and sandbox provider to vary.

The default configuration selects LAB's direct Gym-native loop. Additional
configurations select Gym's Hermes, Claude Code, and Codex agents through:

- `agent_server_module`
- `agent_server_class`
- `agent_config_class`
- `agent_kwargs`

The runner also accepts the policy `model_server`, LAB runtime task and skill
paths, concurrency and layered timeout settings (including
`sandbox_staging_timeout_seconds` for large archive extraction and collection), `sandbox_provider`, an optional
`sandbox_image`, a model URL override, Docker-only network settings,
`opensandbox_request_fraction`, and a result directory. See `configs/` for
complete examples.
Those configurations include the LAB `resources_only.yaml` definition so Gym
starts one resource server and only the selected agent; the combined
compatibility config is reserved for Harbor.

## Execution boundary

For each rollout, the runner validates `instance_id`, resolves a
provider-compatible LAB image, provisions a portable Linux runtime for the
selected harness through that same sandbox provider, and starts an agent-only
sandbox. Docker can build the LAB image automatically as a local convenience;
every other provider receives the configured image reference unchanged.
Runtimes are immutable and content-addressed by their setup script,
requirements, configured CLI pin, LAB image and provider identity, Gym
packaging metadata, and installed `nemo_gym` source. An interprocess lock and
atomic publication prevent concurrent evaluations from rewriting a runtime
that is being built or used.

Dependency provisioning exposes an explicit allowlist containing only the
installer and required Gym package files. It never mounts the repository,
`env.yaml`, task tests, judge credentials, or unrelated files.

Agent source, documents, skills, runtime, and runner inputs are transferred
through Gym's sandbox file API. The sandbox has no writable host mounts.

Only the task instructions and public skill manuals are supplied to the
configured agent. The native choice reproduces upstream LAB's model → function
tools → tool results loop using Gym's Responses API; it uses LAB's canonical
`bash`, `read`, `write`, `write_docx`, `edit`, `glob`, and `grep` tools. Its
results are downloaded to a private temporary directory,
the sandbox is destroyed, and links, devices, traversal paths, and other unsafe
archive entries are rejected before Gym creates host artifacts. A fresh
verifier-only sandbox then receives a separately staged, sanitized `lab-run`
tree, rubric tests, and verifier-only judge credentials. The verifier sees
deliverables at `lab-run/output`, preserving LAB's contract without sharing a
process namespace or writable filesystem with the agent.

The default provider is Docker, but the common path uses only Gym's
`AsyncSandbox` operations: start, upload, execute, download, and stop. Docker,
ECS Fargate, Enroot, Apptainer, OpenSandbox, Daytona, and OpenShell can all be
selected with `sandbox_provider` without changing LAB code. The runtime
builder, agent, and verifier use the same selected provider. No common command
requests root, mounts a host path, invokes a container CLI, interprets the
provider's image format, or assumes the image's configured user. Transfer
archives are made readable to non-root image users, and all large runtime,
scratch, and transfer writes stay under the sandbox's writable `/sandbox`
tree rather than a provider's potentially bounded `/tmp`.
LAB's generated image also includes OpenShell's Docker-driver prerequisites:
`iproute2`, a high-UID `sandbox` identity, and a writable work directory.

Docker's automatic image build and host-network translation are optional
provider conveniences. ECS uses Gym's provider-native SSH reverse tunnel to
expose the rollout-scoped policy proxy as `LAB_POLICY_MODEL_URL`, keeping model
credentials out of the sandbox. Enroot and Apptainer share the orchestrator's
host network. OpenSandbox, Daytona, and OpenShell need a
`sandbox_model_base_url` that is reachable from their remote sandbox whenever
Gym's derived policy-proxy URL is host-local. An explicit
`sandbox_model_base_url` is always used unchanged. A reachable credential-free
Gym proxy is preferred. For a directly authenticated endpoint,
`sandbox_model_api_key_env` may name a launcher environment variable whose
value is copied into the agent sandbox as `LAB_POLICY_API_KEY`. It is not
serialized into runner metadata or supplied to the runtime builder or verifier,
but the evaluated agent can read its own environment. Use this fallback only
with a narrowly scoped, short-lived key and rotate it after the run. Before
importing the selected harness, the in-sandbox runner
performs a bounded, proxy-aware HTTP connectivity check and writes
`runtime/runner_status.json`. Connectivity, harness, sandbox, and verifier
failures set `mask_sample`, skip the judge when applicable, and carry
`_ng_failure_class` so rollout collection sends them to the failure sidecar for
bounded retry. Deterministic task and harness-configuration failures also carry
`_ng_failure_terminal: true` and are not retried. `mask_sample` is a training
hint, not the routing signal. Max-turn and context-limit stops are valid
incomplete outcomes: the runner verifies their partial deliverables and keeps
them in the main rollout JSONL. For Hermes, the runner also disables its optional `/models`
pricing and context-metadata probes; Gym supplies the selected model explicitly,
and actual model-call access logging is enabled.

For OpenSandbox, each LAB task's declared resources are the burst limits. By
default, `opensandbox_request_fraction: 0.25` sets CPU and memory scheduling
requests to 25% of those limits so concurrent evaluations can pack densely.
Disk and any requested GPU are not oversubscribed. The same split is applied to
the runtime builder, agent, and verifier sandboxes. This is the same ratio used
by Gym's
[Mini SWE Agent 2 config](../mini_swe_agent_2/configs/mini_swe_agent_2.yaml),
where 0.5 CPU and 2 GiB are requested against limits of 2 CPU and 8 GiB. Set
the value closer to `1.0` when the cluster needs stronger reservations, or to
`null` to let the OpenSandbox server use one resource map for both requests
and limits.

Codex currently has the largest portable runtime archive. On a remote
OpenSandbox deployment, keep the provider's
`connection.request_timeout_s` at least as high as the runner's 900-second
staging timeout and reduce concurrency if parallel file transfers saturate the
OpenSandbox file API. These deadlines cover infrastructure transfer and
staging; they do not change the model's agent-phase deadline.

Providers that enforce different network access by phase can use
`runtime_builder_provider_options`, `agent_sandbox_provider_options`, and
`verifier_sandbox_provider_options`. For OpenShell, use separate least-privilege
policies: allow only package registries in the runtime builder, only the Gym
policy proxy in the agent, and only the judge endpoint in the verifier. The LAB
image's global Node document libraries are exposed through an explicit
`NODE_PATH`; OpenShell's injected HTTP(S) proxy is trusted only inside its inner
runner so policy-model traffic passes through the gateway policy engine. The
builder's `registry.npmjs.org` endpoint must use `protocol: rest`, `access:
read-only`, `enforcement: enforce`, and `allow_encoded_slash: true`, because npm
uses encoded slashes for scoped packages.

The checked-in configs decode the whole phase-option maps from
`NEMO_GYM_LAB_RUNTIME_BUILDER_PROVIDER_OPTIONS`,
`NEMO_GYM_LAB_AGENT_SANDBOX_PROVIDER_OPTIONS`, and
`NEMO_GYM_LAB_VERIFIER_SANDBOX_PROVIDER_OPTIONS`. This avoids inherited-config
merge restrictions when a benchmark variant needs to add provider-specific
keys such as `policy`; each setting defaults to `{}`.

Artifacts are written below
`results/legal_agent_bench/<harness>_jobs/<model>/<YYYYMMDD-HHMMSS_hash>/<task_name>_<run_id>/`.
For example, the default native loop uses
`results/legal_agent_bench/native_jobs`, while Hermes uses
`results/legal_agent_bench/hermes_jobs`. The model
segment is the configured `policy_model_name`, normalized into one safe path
segment. The task name is also normalized, and the run ID is an eight-character
unique suffix. A session directory is created only when its first rollout starts.
Each trial includes the inner Gym trajectory, harness logs, LAB run
configuration and metrics, completed output files, verifier artifacts, and a
top-level `run_summary.json`. The rollout response also includes direct paths
for the summary, trajectory, stdout, stderr, output directory, and verifier
report.

Set `NEMO_GYM_LAB_RESULTS_DIR` to redirect the artifact root. This is useful
when Gym runs inside a Linux VM: use the VM's native filesystem instead of a
macOS-shared mount to avoid cross-OS permissions and ownership behavior.

For setup, five copy-paste smoke commands, and result inspection, see the
[benchmark README](../../benchmarks/legal_agent_bench/README.md#test-the-various-harnesses).
Run `gym` from an activated repository environment rather than prefixing
server-starting commands with `uv run`.

## Adding a harness

A harness can be selected without changing this runner when it:

1. is implemented as a Gym Responses API agent;
2. can run inside the LAB Linux sandbox image;
3. has a portable dependency script at
   `responses_api_agents/<agent>/scripts/<agent>_deps.sh`; and
4. can use the configured OpenAI-compatible policy endpoint.

Add a standalone configuration under `configs/`, then add a benchmark variant
that inherits it. CLI versions belong in the harness configuration's
`agent_kwargs`; provisioning derives its exact package specification from that
pin and rejects unpinned Claude Code or Codex configurations.

Docker is the zero-configuration local backend. See the benchmark README for
the provider matrix, image and networking requirements, lifecycle smoke test,
and one-task override examples. The separate Harbor compatibility variant does
not use this runner and is Docker-only.
