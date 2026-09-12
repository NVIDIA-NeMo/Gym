# mini-SWE-agent sandboxed harness

An asyncio port of mini-SWE-agent 2.4.6's default agent, Responses API model,
and local shell environment. The pinned package supplies the prompts, native bash
tool schema, action parser, and observation formatter. The model loop runs on the
Gym host; commands execute in the task sandbox created by the resources server.

The initial implementation preserves the collected H3 harness's reasoning replay
option, full raw command outputs, upstream-format trajectory archive, and episode
affinity header. It removes run-specific imports and global provider monkey patches.

Use the compositions at:

- `/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/gym_workdir/gym_tb4/benchmarks/terminal_bench_4/mini_swe_agent.yaml`
- `/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/gym_workdir/gym_tb4/benchmarks/terminal_bench_2_1/mini_swe_agent.yaml`

TB4 delegates grading to `terminal_bench_4`: collect declared artifacts, stop the
agent sandbox, start a fresh verifier sandbox, restore artifacts, and run its tests.
The harness never receives the verifier's tests. A row's positive finite
`agent_timeout_sec` overrides the configured episode timeout; TB4 defaults to
28,800 seconds. Reaching the deadline still invokes verification. Connection or
verifier failures cancel the cookie-scoped seeded TB4 session and stop the agent.

Defaults: 500 steps, 30 seconds per shell command, `/bin/sh`, no cost limit, no
compaction. `replay_reasoning_items` controls what the model sees on subsequent
turns; reasoning is always preserved in captured model responses. Set
`dump_trajectory_dir` outside the checkout to retain the self-contained full trace.

Optional `sandbox_hostname_suffix: "-0"` checks the Kubernetes hostname in the same
exec as every command. Enable it only for OpenSandbox deployments that use that
hostname convention. Mismatches are ungraded infrastructure outcomes and skip
verification. This detects stale endpoints; it does not enforce offline networking.

For offline training, put `network_policy: {defaultAction: deny, egress: []}` in
the **resources server's** `sandbox_config.provider_options`. Mini-SWE needs no
model endpoint exception because model requests originate on the Gym host. The
policy applies to TB4 agent and verifier sandboxes. Required task and verifier
dependencies must already be available offline.

The `mini-swe-gym-episode-v2` export captures the first prepared model request
before dispatch. Its complete input defines the prompt boundary dynamically;
the exported output contains the subsequent episode history, including later
user feedback. Bash tools and explicit request settings come from that capture.
SDK defaults are marked separately from observed parameters. Both `/run` and
`/responses` expose the normalized request and export metadata. Seeding retains
the original task request. The archive keeps that original request, the complete
trajectory, raw model responses, shell records and the captured request hash.
No-call and inconsistent-prefix outcomes are explicitly invalid for normalized
training; a failed or empty first response can still have a valid observed prompt.
The flat episode is audit history, not exact model-context replay: format-error
recovery may omit a failed generation from later requests. Provider overrides
must be recorded separately by the launching recipe.

Validation before the export repair: 79 focused tests passed with 97.14% coverage. Live offline qualification
passed all six mini-SWE model/workflow combinations, including GLM5.3 and Super3.5
on CMH plus NVIDIA inference, each with full capture and scoped cleanup. A native
TB4 task also exercised artifact transfer and completed grading. Results and the
consolidated twelve-case recipe matrix are tracked at
`/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/work/logbook/problems/P260909-tb-tb4-climb-passk/experiments/H7-unified-gym-recipes/runs/2026-09-11_r1-unified-gym-recipes/run.md`.
