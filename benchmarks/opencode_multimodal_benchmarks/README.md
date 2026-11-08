# Visual benchmark catalog

These entries share dataset preparation and a stateless verifier. They can be
used independently of OpenCode:

- [VStar](vstar/README.md)
- [HRBench8K](hrbench8k/README.md)
- [VTCBench](vtcbench/README.md)
- [Agent-X GoalAcc](agentx/README.md)
- [ZeroBench](zerobench/README.md)

Each entry documents its dataset and measurement contract. The catalog defaults
to `simple_agent` with no external task tools. Compose [`opencode_sessions.yaml`](opencode_sessions.yaml) after a catalog config
to use the OpenCode multimodal session adapter. Configure the sandbox provider and
model separately. Existing compatibility configurations remain available in
[ONBOARDING.md](ONBOARDING.md).

## Run configuration

Supply a policy model using Gym's [model configuration guide](../../fern/versions/latest/pages/model-server/index.mdx).
For judged tasks, explicitly configure the shared resources-server instance:

```yaml
opencode_multimodal_benchmarks:
  resources_servers:
    opencode_multimodal_benchmarks:
      judge_model: <judge-model-id>
      judge_server: visual_judge
```

Also configure the `visual_judge` model-server instance and its runtime credentials.
This route captures judge calls with Gym's rollout correlation. Alternatively,
set `judge_endpoint` and `judge_api_key_env` for direct HTTP judging. The catalog
does not select a hosted model or endpoint. ZeroBench requires no judge.

The shared verifier defaults to a 120-second timeout and three transport attempts
per judge request. Agent-X uses 300 seconds and ten attempts. VTCBench additionally
uses the public five-verdict parsing schedule. Judge failures need resolution
before a complete score is reported.

For each evaluation, retain the resolved config and inspect rollout health,
coverage and judge failures. Enable model-call capture when collecting evidence.
A valid manifest and passing verifier fixtures do not establish score parity or
harness conformance.

## Grading and recovery

The Resources Server owns grading. Deterministic scores, including incorrect or
empty-answer zeros, use `observe_programmatic_grading`. Gym-routed judges use
`observe_llm_judge_grading`, `call_judge`, and a parsed-verdict marker. The recorded
operation binds the saved policy response and reward to the judge reply. Use
`judge_server` and model-call capture for qualification. Direct HTTP judging is
supported for compatibility, but its unobserved judge calls cannot establish a
fully observed grading record. It is never labeled model-free grading.

An invalid judge reply is not a model mistake. Transport or parsing failures
preserve the policy response and become masked `judge_failed` outcomes; recover
by rejudging the saved answer. If a parsing retry succeeds after an unparseable
reply, the score is retained but the grading operation remains `unknown`: an
invalid reply is not falsely marked as a parsed verdict. Inspect these outcomes
and rejudge before claiming complete grading evidence.

Catalog pairings have no whole-episode deadline by default. To set one, configure
`visual_environment.environment_servers.single_agent_turn.default_episode_timeout_seconds`.
`BaseEnvironmentServer.run_request` enforces it with `asyncio.timeout` on the
event loop's monotonic clock. Cleanup has a separate
`cleanup_timeout_seconds` budget and runs after the episode deadline.

The OpenCode overlay has a 131072-token context/output ceiling and a 55+10 turn
budget. Its `timeout` setting (optionally narrowed by request metadata
`agent_timeout_sec`) is a separate harness execution deadline, enforced
by the shared OpenCode sandbox session using the monotonic clock; installation
uses `sandbox_install_timeout_seconds`, and close uses
`session_close_timeout_seconds`. The judge timeout above bounds each transport
attempt (including Gym transport), not the wait for the judge concurrency slot.
Scheduler walltime bounds one allocation; the resumable chain budget bounds all
allocations. Save each explicitly in the resolved run configuration.

For a submitted qualification run, set the existing run-level recovery and
delivery controls under your benchmark in `driver.benchmarks`:

```yaml
resumable:
  max_retries: 3
  max_walltime: "16:00:00"
  recovery_outcome_policy:
    by_failure_kind:
      agent_timeout: fail
      model_context_limit: fail
    default_action: retry
run:
  mlflow_tracking_required: false
```

This example deliberately keeps delivery local; a tracked qualification must set
`mlflow_tracking_required: true` and retain its export receipt. Neither setting
relaxes evaluation-health checks. The configured agent/model limits are final outcomes;
infrastructure faults may be retried within the saved budget, and judge faults
use saved-answer rejudging. Do not use recovery to give a settled wrong answer
another policy attempt.

Dataset bytes, repeats, model, prompt, sampling, judge and scoring settings are
material resume identity. Placement addresses and the collector's
`dispatch_connect_grace_s` are operational. Keep the original run directory:
resume validates its manifest and leaves settled rows intact. Never override a
material mismatch with `allow_unsafe_resume`; start a new run instead.

Before qualification, resolve and dry-run the submission, then run three distinct
tasks and the fault/cut-and-resume fixtures described in Gym's
[benchmark skill](../../.agents/skills/add-benchmark/SKILL.md#step-7b-failure-handling-conformance).
Unit fixtures establish code contracts; they do not replace a real-model run or
the submitted run's health and recovery gates.

## Dataset identity

VStar/HRBench8K, Agent-X and ZeroBench are checked against the pinned manifests or
source checksum. VTCBench TSV releases are caller-supplied: pass
`prepare_script_args.sha256` for qualification and retain the preparation's
`dataset.json` with its source and prepared-task hashes. Use a separate output
path for a different release or preprocessing policy. A caller-supplied variant
must not be reported as the pinned reference run without matching these hashes.

Prepared VStar, HRBench8K and VTCBench image inputs use embedded data URLs.
Hosted model endpoints need no access to the preparation machine’s image paths.
