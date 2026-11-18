# Example: IFEval

Moved from section 4 of [SKILL.md](../SKILL.md), with semicolons replaced by commas or periods and no other change.

`benchmarks/ifeval/` needs no judge and no credentials. Its `config.yaml` chains to
`resources_servers/instruction_following/configs/instruction_following.yaml`, and its `prepare.py`
downloads the prompts from the upstream default branch, so the Gym commit does not pin the data: compare
the runs' materialized inputs when you compare two runs. Its block is:

```yaml
driver:
  benchmarks:
    ifeval:
      prepare:
        config_paths: [benchmarks/ifeval/config.yaml]
      run:
        config_paths: [benchmarks/ifeval/config.yaml]
        split: benchmark
      resumable:
        max_retries: 3
        max_walltime: "08:00:00"
```

The rest is section 2. A policy that reasons by default and is served with `--reasoning-parser` (Qwen3,
for example) needs no section 3 keys, because the shipped `vllm_model` client already expects the parser.
A model that needs them adds them to the same `run` mapping. In the `--dry-run` script the job runs
`gym eval prepare '+config_paths=[benchmarks/ifeval/config.yaml]'`, then `gym eval run --model-type
vllm_model` with the same `+config_paths`, `+split=benchmark` and the three policy keys submit filled in.
Validate it (section 5) and smoke it (section 6), then hand the file to run-and-babysit-eval.
