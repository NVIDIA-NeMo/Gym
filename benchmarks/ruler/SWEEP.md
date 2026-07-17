# Multi-length RULER sweeps

This benchmark flavor prepares several RULER sequence lengths as one Gym
dataset. Each length still uses the existing single-length preparation helper;
the sweep adapter only concatenates its outputs and adds `sequence_length`,
`sequence_length_tokens`, and per-subset `source_index` metadata.

```bash
gym eval prepare --benchmark ruler/config_sweep \
  "+prepare_script_args.model='/path/to/checkpoint-or-tokenizer'" \
  "+prepare_script_args.lengths=[8192,16384,32768,65536,131072,262144]"
```

Run the combined dataset like any other benchmark:

```bash
gym eval run \
  --benchmark ruler/config_sweep \
  --model-type vllm_model \
  --model-url http://localhost:8000/v1 \
  --model-api-key dummy \
  --model /path/to/checkpoint \
  --output results/ruler_sweep.jsonl \
  --split benchmark \
  --temperature 0 \
  --top-p 1 \
  --resume \
  ++overwrite_metrics_conflicts=true \
  ++reuse_existing_data_preparation=true
```

Additional preparation arguments are forwarded unchanged to the single-length
helper. This keeps sequence-length composition independent of prompt format,
tokenizer choice, or other RULER preparation flavors.
