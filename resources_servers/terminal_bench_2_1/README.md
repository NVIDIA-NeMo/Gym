# Description

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
