# LMArena proxy v4

LMArena proxy v4 benchmark (2,328 prompts), Gemini 3.8 Flash judge, and upper-length-only
style control.

Follow the [Arena setup](../../resources_servers/arena/README.md#setup), then run:

```bash
gym eval prepare --config benchmarks/lmarena_v4/config.yaml

gym eval run \
  --config benchmarks/lmarena_v4/config.yaml \
  --agent lmarena_v4_benchmark_agent \
  --split benchmark \
  --output results/lmarena_v4/my-model/rollouts.jsonl \
  --concurrency 32 \
  --resume \
  ++reuse_existing_data_preparation=true
```

Add `--limit 2` for a smoke test.
