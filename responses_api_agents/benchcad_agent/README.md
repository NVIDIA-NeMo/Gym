# BenchCAD agent

Evaluates BenchCAD with Gym's sandboxed OpenCode harness and pinned upstream
prompts and scorers. Agent execution and prediction execution use separate
sandboxes; reference geometry and answers stay in the trusted scorer.

See the [BenchCAD evaluation guide](../../fern/versions/latest/pages/evaluation-tutorials/benchcad.mdx)
for data preparation, Docker setup, model configuration, and scoring.
The benchmark configuration is [benchmarks/benchcad/config.yaml](../../benchmarks/benchcad/config.yaml).

Run component tests with:

```bash
gym env test +entrypoint=responses_api_agents/benchcad_agent +should_validate_data=true
```

Tests exercising upstream scoring require a prepared, pinned BenchCAD checkout.
Set `BENCHCAD_TEST_ROOT` to its absolute path if it is outside
`benchmarks/benchcad/.cache/upstream`; those tests skip when it is unavailable.
