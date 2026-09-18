# harness_agent

Runs any gym agent harness in a sandbox with any resources server.

## Per-task metadata keys

Task shape lives in the dataset rows to remain agnostic, not the agent config.
Reserved keys in `responses_create_params.metadata`:

| Key | Behavior when present |
|---|---|
| `docker_image` | sandbox image for the task (else the `sandbox_image` default) |
| `workdir` | in-box dir the agent's `repo_dir` points at, so edits land in the graded tree |
| `sandbox_eval` | JSON grading spec run in the box right after the solve, reward goes in response metadata as `sandbox_reward` (the spec is stripped from the agent's request so it cannot peek at tests) |
Tasks with an external verifier (e.g. math) need none of these beyond an image.
This is largely for swe bench now.

## Offline scientific benchmarks (development)

Compose the benchmark with one harness preset, for example in a local config:

```yaml
config_paths:
- benchmarks/apex_shortlist/harness.yaml
- responses_api_agents/harness_agent/configs/opencode.yaml
```

Use `benchmarks/hle/harness.yaml` for HLE, or select `configs/pi.yaml` instead of
`configs/opencode.yaml` to exercise Pi. Select one preset per configuration.
Both benchmarks retain their existing preparation, grading and repeats. HLE
uses its Explanation/Answer/Confidence instructions in the user message. The
harness presets add only the network-availability note to the native agent setup.

Set `HARNESS_SANDBOX_IMAGE` to an image digest containing the harness binaries,
scientific tools, and an isolated Gym interpreter at `/opt/gym-runtime/bin/python`.
See [the runtime image](image/README.md). Sandbox access still uses the normal
OpenSandbox provider configuration. The model server must advertise an address
reachable from the sandbox; a workstation address is not necessarily routable
from the assigned Kubernetes cell.

`network_access: model_only` enforces an OpenSandbox deny-by-default policy with
only the Gym model-server host allowed. It overrides permissive network settings,
rejects unsupported providers, and rejects resource-owned sandboxes whose policy
cannot be checked. Backend API keys stay in the host-side Gym model server. An
explicit `sandbox_model_base_url` must also point to a reachable model proxy.

The shared runner carries rollout identity and returns native agent observations.
It supplies `resolved_model_base_url` as a runtime input to the inner agent,
including `/v1` and any rollout/capture path. Agents using the shared URL resolver
use that endpoint verbatim; ordinary server-config resolution is unchanged.
`artifacts_dir` saves generation receipts and runner logs before host-side grading;
receipt directories hash the rollout ID to avoid interpreting dataset values as
paths. `execution_failure_reward_zero` skips grading completed harness failures
and records reward zero with `harness_failed=true`. Setup, export and judge errors
still raise; use Gym's failure sidecar to keep unrelated rows running, and account
for those missing rows before reporting full benchmark coverage.

The OpenCode preset uses the remaining-context plugin, disables compaction, and
sets 400 steps. Pi's preset is for plumbing validation: its native token-budget
and compaction behavior has not yet been aligned or benchmarked. All enclosing
benchmark rollout limits are four hours; individual tools retain native limits.

The `apex_shortlist/opencode` and `hle/opencode` benchmark entrypoints compose
these presets. `hle/opencode_search` adds the existing Tavily resource with
`network_access: model_and_tools`. The shared runner seeds each `tool_servers`
entry through Gym and passes its signed per-session MCP headers to OpenCode.
Only the configured model and tool hosts are allowed; Tavily API keys remain in
the resource process. Remote MCP injection currently supports OpenCode; other
harnesses are rejected when tool servers are configured.

Tavily enforces the HLE domain/URL exclusions. Deployments can supply an additional
policy file to the resource without changing the public benchmark. See
[the Tavily resource](../../resources_servers/tavily_search/README.md) for the
three exposed tools and policy behavior.
