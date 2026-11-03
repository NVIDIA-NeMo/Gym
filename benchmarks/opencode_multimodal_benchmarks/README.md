# Visual benchmark catalog

These entries share dataset preparation and a stateless verifier. They can be
used independently of OpenCode:

- [VStar](vstar/README.md)
- [HRBench8K](hrbench8k/README.md)
- [VTCBench](vtcbench/README.md)
- [BabyVision](babyvision/README.md)
- [Agent-X GoalAcc](agentx/README.md)
- [ZeroBench](zerobench/README.md)

Each entry documents its dataset and measurement contract. The catalog defaults
to `simple_agent` with no external task tools. Compose [`opencode_sessions.yaml`](opencode_sessions.yaml) after a catalog config
to use the OpenCode multimodal session adapter. Configure the sandbox provider and
model separately. Existing compatibility configurations remain available in
[ONBOARDING.md](ONBOARDING.md).

## Run configuration

Supply a policy model using Gym's [model configuration guide](../../../fern/versions/latest/pages/model-server/index.mdx).
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
