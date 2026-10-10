# terminal_bench_2_1

TODO: Describe this benchmark and replace the sample data.

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_2_1`

## Harbor task shape

The same 89 tasks also run through the generic `harbor` resources server, with the task folders read
directly instead of a JSONL row per task. Materialize the pinned fork once into the shared datasets folder:

```bash
python benchmarks/terminal_bench_2_1/prepare_harbor_taskset.py
```

This writes `datasets/terminal-bench-2-1/<task>/` for every task, applies the `tests/test.sh` and
`solution/solve.sh` fixes the `terminal_bench_2_1` server applies at run time as plain files, and records
the commit and the patched tasks in `manifest.toml`.

Check the reference solutions, then run a model:

```bash
gym dataset validate datasets/terminal-bench-2-1
gym eval run datasets/terminal-bench-2-1 --agent terminus_2_sandboxed_agent \
  --model-type openai_model --model <model> --model-url <url>
```

To match the `terminal_bench_2_1` server's sandbox sizing for a like-for-like comparison, add an
overlay that sets `sandbox_resources_override` on the generated `harbor_terminal-bench-2-1_resources_server`
block (see `resources_servers/harbor/configs/harbor.yaml` for the knobs).
