# Description

Data links: ?

# Quickstart
## Apply golden patches
### Start resources server
```bash
gym env start \
    --config resources_servers/swebench/configs/swebench.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +swebench_resources_server.resources_servers.swebench.is_verifying_golden_patch=true
```

### One golden patch smoke test
In a separate terminal:
```bash
# If you haven't already, prepare SWE Bench benchmark data
gym eval prepare --config benchmarks/swebench/verified/config.yaml

python resources_servers/swebench/client.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl
```

### Full SWE Verified golden patch smoke test
In a separate terminal:
```bash
python resources_servers/swebench/apply_golden_patch.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    +limit=...  # No limit for full samples
```


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
