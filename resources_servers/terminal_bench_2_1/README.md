# Description

> [!WARNING]
> **Legacy.** Terminal-Bench 2.1 now runs on the [`harbor` environment server](../../environment_servers/harbor/README.md)
> from the task folders of [`environments/harbor/terminal_bench_2_1`](../../environments/harbor/terminal_bench_2_1/README.md):
> `gym eval run --benchmark terminal_bench_2_1/terminus_2` or `terminal_bench_2_1/opencode`. This server still backs
> `terminal_bench_2_1/nooa`, `terminal_bench_2_1/inkling_small` and the `*_legacy_agent` recipes, and will be retired
> once they move. Do not build new benchmarks or harness pairings on it; see
> [the benchmark README](../../benchmarks/terminal_bench_2_1/README.md#differences-from-the-resources-server-path)
> for how its scores differ from the Harbor path's.

Data links: ?

# Quickstart
## Apply golden patches
### Start resources server
```bash
gym env start \
    --config resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.debug=true \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.is_verifying_golden_patch=true
```

### Full Terminal Bench 2.1 golden patch smoke test
In a separate terminal:
```bash
gym eval prepare --config benchmarks/terminal_bench_2_1/terminus_2_legacy_agent.yaml

python resources_servers/terminal_bench_2_1/apply_golden_patch.py \
    +benchmark_jsonl=benchmarks/terminal_bench_2_1/data/benchmark.jsonl \
    +limit=...  # No limit for full samples
```

Expected golden patch resolve rates:
```bash

```


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
