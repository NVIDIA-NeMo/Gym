# SWE-bench Pro

This resources server evaluates patches for the public
[ScaleAI SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) benchmark with NeMo Gym
`AsyncSandbox` providers.

The integration follows the evaluator from
[`scaleapi/SWE-bench_Pro-os`](https://github.com/scaleapi/SWE-bench_Pro-os/tree/ca10a60a5fcae51e6948ffe1485d4153d421e6c5):
it starts the task image identified by `dockerhub_tag`, resets `/app` to the base commit, applies the candidate
patch, runs the task-specific script, parses its JSON output, and requires every fail-to-pass and pass-to-pass
test to pass.

## Prepare data

For a Hermes rollout using this resources server, see the
[Hermes Environment Server guide](../../fern/versions/latest/pages/evaluation-tutorials/hermes-swe-bench-pro.mdx).
The agent uses the same task preparation and verifier as OpenCode. Command-based
agents can request `create_pty=false` at `/seed_session`; the default still creates
a terminal. Connectable providers also return `sandbox_descriptor`, and agents
can release an abandoned task through the cookie-scoped `/close_session` endpoint.

Optional local SIFs must have provenance manifests produced by
`python -m resources_servers.swebench_pro.image_cache`. Set `image_template` to
`/cache/sifs/{image_digest_hex}.sif`; the helper names files by the pinned digest.
Before starting a container, the server checks the original registry URI and
the SIF checksum against the manifest. Seed and verify responses include
`image_provenance`; the registry path remains unchanged when no template is set.

```bash
uv run python benchmarks/swebench/pro/prepare.py
```

Preparation pins and embeds the task-specific run scripts, parsers, and Dockerfile metadata from the upstream
evaluator commit. The resources server therefore does not access GitHub while serving verification requests.

## Migrate standalone sandboxed agents

Use this workflow when replacing an adapter that manages task setup, execution, verification,
and cleanup with a harness that runs through Environment Server agent sessions. Migration is
specific to each harness: confirm that its replacement supports the required lifecycle and
benchmark capabilities before changing a working configuration. A `*_sandboxed_agent` name
alone does not mean the component has been deprecated.

1. **Check the replacement's capabilities.** Confirm agent-session support, accepted task
   input, required tools and MCP grants, model routing, and execution limits. Check the
   harness documentation for supported sandbox providers and worker/concurrency restrictions.
2. **Route execution through the Environment Server.** Configure its Resources Server and
   agent references. Include the composed configuration in both server startup and evaluation
   commands, including `--no-serve` runs. Direct HTTP callers use the Environment Server's
   `/run`; changing the agent name in an existing agent `/run` call does not create a session.
   Use the task format expected by the environment; `single_agent_turn_legacy` accepts
   compatible prepared flat rows through the session lifecycle.
3. **Preserve sandbox ownership.** For tasks where Resources prepares the workspace and
   verifies its state, the agent must connect to that same sandbox. Configure agent-owned
   sandbox creation only when both the harness and the environment support it. Check model
   and tool endpoint reachability from the sandbox and retain required mounts and workdirs.
4. **Map configuration explicitly.** Move harness settings into the replacement agent's
   config namespace and provisioning settings into the appropriate provider or Resources
   Server config. Check runtime pins, architecture/libc, offline installation, and secrets
   delivery. Reconcile per-call token limits, total-response budgets, command timeouts, and
   episode deadlines; similarly named settings can enforce different limits.
5. **Validate execution and reporting.** Run representative real-model tasks, inspect tool
   actions and verifier results, and confirm agent close precedes verification and owned
   resources are cleaned up. Update consumers of adapter-specific artifacts and metrics to
   the new [rollout evidence](https://docs.nvidia.com/nemo/gym/main/observability/rollout-evidence) and outcome format.
   Compare failure coverage as well as rewards; changed runtimes or limits require a new
   benchmark baseline.
6. **Retire the old integration after migration.** Update benchmark recipes, allowed-agent
   lists, manifests, deployment scripts, imports, and documentation before removing an
   adapter. Keep runtime packaging and shared helpers that still have consumers.

For a concrete migration, see the
[Hermes-specific configuration and runtime changes](../../responses_api_agents/hermes_agent/README.md#migrate-from-the-standalone-sandboxed-agent)
and the [SWE-Pro recipe](../../fern/versions/latest/pages/evaluation-tutorials/hermes-swe-bench-pro.mdx).

## Migrate an existing SWE-Pro configuration

Follow the [shared sandboxed-agent migration workflow](#migrate-standalone-sandboxed-agents)
and the [Hermes-specific runtime, request, and reporting changes](../../responses_api_agents/hermes_agent/README.md#migrate-from-the-standalone-sandboxed-agent).
For this benchmark:

- Replace the old agent configuration with `benchmarks/swebench/pro/hermes.yaml`, and
  select `swebench_pro_hermes_agent` when using `--no-serve`.
- Move agent settings under `swebench_pro_hermes_agent.responses_api_agents.hermes_agent`.
  Move SWE-Pro `image_template` and image-cache provenance settings under
  `swebench_pro_hermes_resources_server.resources_servers.swebench_pro`.
- Reuse prepared SWE-Pro JSONL rows through `single_agent_turn_legacy`; no data conversion
  is needed. The SWE-Pro verifier is unchanged by the removal.

## Golden-patch smoke test

Start the resources server with an OpenSandbox provider:

```bash
gym env start \
  --config resources_servers/swebench_pro/configs/swebench_pro.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  +swebench_pro_resources_server.resources_servers.swebench_pro.is_verifying_golden_patch=true
```

In another terminal, verify one row:

```bash
python resources_servers/swebench_pro/client.py \
  +benchmark_jsonl=benchmarks/swebench/data/swebench_pro_benchmark.jsonl
```

Run a bounded batch by setting both the row limit and sandbox concurrency:

```bash
python resources_servers/swebench_pro/apply_golden_patch.py \
  +benchmark_jsonl=benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  +limit=10 \
  +concurrency=2
```

The upstream evaluator and bundled task scripts are MIT licensed. NeMo Gym's adapter code is Apache-2.0.
