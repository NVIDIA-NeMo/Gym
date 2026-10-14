# BenchCAD agent

Evaluates BenchCAD with one direct model response or Gym's sandboxed OpenCode
harness, using pinned upstream prompts and scorers. Generated CAD executes in
an isolated prediction sandbox; reference geometry and answers stay in the
trusted scorer. OpenCode remains the default.

See the [BenchCAD evaluation guide](../../fern/versions/latest/pages/evaluation-tutorials/benchcad.mdx)
for data preparation, Docker setup, model configuration, and scoring.
Use [config.yaml](../../benchmarks/benchcad/config.yaml) for OpenCode or
[single_turn.yaml](../../benchmarks/benchcad/single_turn.yaml) for direct inference.

Run component tests with:

```bash
gym env test +entrypoint=responses_api_agents/benchcad_agent +should_validate_data=true
```

Tests exercising upstream scoring require a prepared, pinned BenchCAD checkout.
Set `BENCHCAD_TEST_ROOT` to its absolute path if it is outside
`benchmarks/benchcad/.cache/upstream`; those tests skip when it is unavailable.
