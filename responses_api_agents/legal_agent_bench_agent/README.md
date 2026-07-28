# Legal Agent Bench Configurable Agent

This task-driven Gym agent runs a configured Gym Responses API agent inside the
Legal Agent Bench (LAB) Docker environment. It keeps the benchmark task set,
skills, and verifier fixed while allowing the agent harness to vary.

The committed configurations select Gym's Hermes, Claude Code, and Codex agents
through:

- `agent_server_module`
- `agent_server_class`
- `agent_config_class`
- `agent_kwargs`

The runner also accepts the policy `model_server`, LAB runtime task and skill
paths, concurrency and timeout settings, Docker network and model URL
overrides, and a result directory. See `configs/` for complete examples.
Those configurations include the LAB `resources_only.yaml` definition so Gym
starts one resource server and only the selected agent; the combined
compatibility config is reserved for Harbor.

## Execution boundary

For each rollout, the runner validates `instance_id`, builds or reuses the
content-addressed LAB image, provisions a cached portable Linux runtime for the
selected harness inside that image, and starts an agent-only Docker sandbox.
Provisioning selects ARM64 or x86-64 dependencies from the container
architecture, so the host does not need to execute container binaries. The
runtime cache is invalidated by changes to its setup script, requirements,
configured CLI pin, LAB image, Gym packaging metadata, or installed `nemo_gym`
source. Only the selected Gym agent package, source documents, skills, and
portable dependencies are mounted read-only; the repository and LAB task cache
are not mounted. Output and scratch directories are writable.

Only the task instructions and public skill manuals are supplied to the
configured agent. After it exits, its container is destroyed before a fresh
verifier-only sandbox starts. That sandbox receives the completed `lab-run`
directory read-only, then receives the rubric tests and judge credentials.
The verifier sees deliverables at `lab-run/output`, preserving LAB's contract
without sharing a process namespace with the agent.

The runner translates derived host-loopback model URLs to
`host.docker.internal` on Docker Desktop and registers the equivalent host
gateway for Linux bridge networking. An explicit `sandbox_model_base_url` is
used unchanged. Before importing the selected harness, the in-container runner
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
[benchmark README](../../benchmarks/legal_agent_bench/README.md#smoke-test-every-harness).
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

The runner is compatible with Docker only.