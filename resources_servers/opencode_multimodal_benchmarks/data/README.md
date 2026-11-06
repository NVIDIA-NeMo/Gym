# Synthetic examples

These five color-square questions are generated fixtures, not benchmark samples.
Their embedded PNGs need no downloads. `example_metrics.json` is produced by
Gym's dataset validator; `example_rollouts.jsonl` contains real-model responses
through the OpenCode session adapter, with machine-local capture paths omitted.
The raw captures remain outside the source tree. These examples check wiring and
verification, not benchmark accuracy.

Validate the dataset with:

```bash
gym dataset collate \
  '+config_paths=[resources_servers/opencode_multimodal_benchmarks/configs/example.yaml]' \
  +output_dirpath=/tmp/visual-examples +mode=example_validation
```

For OpenCode execution, compose `example.yaml` with
`benchmarks/opencode_multimodal_benchmarks/opencode_sessions.yaml`, a model
configuration and a sandbox provider, as described in the agent README.
