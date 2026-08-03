# Legal Agent Bench Configurable Agent

This task-driven Gym agent runs a configured Gym Responses API agent inside a
Legal Agent Bench (LAB) sandbox. It keeps the benchmark task set, skills, and
verifier fixed while allowing the agent harness and sandbox provider to vary.

The committed configurations select Gym's Hermes, Claude Code, and Codex agents
through:

- `agent_server_module`
- `agent_server_class`
- `agent_config_class`
- `agent_kwargs`

The runner also accepts the policy `model_server`, LAB runtime task and skill
paths, concurrency and timeout settings, `sandbox_provider`, an optional
`sandbox_image`, Docker platform, network and model URL overrides, and a result
directory. See `configs/` for complete examples.
Those configurations include the LAB `resources_only.yaml` definition so Gym
starts one resource server and only the selected agent; the combined
compatibility config is reserved for Harbor.

## Execution boundary

For each rollout, the runner validates `instance_id`, builds or reuses the
content-addressed LAB image, provisions a portable Linux runtime for the
selected harness, and starts an agent-only sandbox. Runtimes are immutable and
content-addressed by their setup script, requirements, configured CLI pin, LAB
image identity and architecture, Gym packaging metadata, and installed
`nemo_gym` source. An interprocess lock and atomic publication prevent
concurrent evaluations from rewriting a runtime that is being built or used.

Dependency provisioning exposes an explicit allowlist containing only the
installer and required Gym package files. It never mounts the repository,
`env.yaml`, task tests, judge credentials, or unrelated files.

Agent source, documents, skills, runtime, and runner inputs are transferred
through Gym's sandbox file API. The sandbox has no writable host mounts.

Only the task instructions and public skill manuals are supplied to the
configured agent. Its results are downloaded to a private temporary directory,
the sandbox is destroyed, and links, devices, traversal paths, and other unsafe
archive entries are rejected before Gym creates host artifacts. A fresh
verifier-only sandbox then receives a separately staged, sanitized `lab-run`
tree, rubric tests, and verifier-only judge credentials. The verifier sees
deliverables at `lab-run/output`, preserving LAB's contract without sharing a
process namespace or writable filesystem with the agent.

The default provider is Docker. The runner translates derived host-loopback
model URLs to `host.docker.internal` on Docker Desktop and registers the
equivalent host gateway for Linux bridge networking. A named Gym provider can
be selected through `sandbox_provider`; non-Docker providers require an
immutable registry `sandbox_image` and an explicit reachable
`sandbox_model_base_url`. Before importing the selected harness, the
in-container runner
performs a bounded TCP connectivity check and writes
`runtime/runner_status.json`. Connectivity or harness failures set the
`mask_sample` flag on the row, skip the judge, and are distinct from an ordinary zero-reward
model result. For Hermes, the runner also disables its optional `/models`
pricing and context-metadata probes; Gym supplies the selected model explicitly,
and actual model-call access logging is enabled.

Artifacts are written below
`results/legal_agent_bench/<harness>_jobs/<model>/<YYYYMMDD-HHMMSS_hash>/<task_name>_<run_id>/`.
For example, Hermes uses `results/legal_agent_bench/hermes_jobs`. The model
segment is the configured `policy_model_name`, normalized into one safe path
segment. The task name is also normalized, and the run ID is an eight-character
unique suffix. A session directory is created only when its first rollout starts.
Each trial includes the inner Gym trajectory, harness logs, LAB run
configuration and metrics, completed output files, verifier artifacts, and a
top-level `run_summary.json`. The rollout response also includes direct paths
for the summary, trajectory, stdout, stderr, output directory, and verifier
report.

For setup, four copy-paste smoke commands, and result inspection, see the
[benchmark README](../../benchmarks/legal_agent_bench/README.md#test-the-various-harnesses).
Run `gym` from an activated repository environment rather than prefixing
server-starting commands with `uv run`.

## Adding a harness

A harness can be selected without changing this runner when it:

1. is implemented as a Gym Responses API agent;
2. can run inside the LAB Linux container;
3. has a portable dependency script at
   `responses_api_agents/<agent>/scripts/<agent>_deps.sh`; and
4. can use the configured OpenAI-compatible policy endpoint.

Add a standalone configuration under `configs/`, then add a benchmark variant
that inherits it. CLI versions belong in the harness configuration's
`agent_kwargs`; provisioning derives its exact package specification from that
pin and rejects unpinned Claude Code or Codex configurations.

Docker remains the zero-configuration local backend. Non-Docker providers
require an immutable registry image and a model URL reachable from the sandbox.
